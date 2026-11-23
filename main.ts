import * as fs from 'node:fs';
import * as path from 'node:path';

type Meta = [imm: number, pop: number, push: number];

const OPS: Record<number, Meta> = {
  0: [0, 0, 0], 1: [1, 0, 1], 2: [4, 0, 1], 3: [0, 1, 0], 4: [0, 1, 2], 5: [0, 2, 2], 6: [0, 2, 3], 7: [0, 3, 3],
  16: [0, 2, 1], 17: [0, 2, 1], 18: [0, 2, 1], 19: [0, 2, 1], 20: [0, 2, 1], 21: [0, 2, 1], 22: [0, 2, 1], 23: [0, 1, 1],
  24: [0, 2, 1], 25: [0, 2, 1], 26: [0, 2, 1], 27: [0, 1, 1], 28: [0, 2, 1], 29: [0, 2, 1], 30: [0, 2, 1],
  32: [0, 2, 1], 33: [0, 2, 1], 34: [0, 2, 1], 48: [2, 0, 0], 49: [2, 1, 0], 50: [2, 1, 0], 51: [2, 0, 0], 52: [0, 0, 0],
  64: [0, 1, 1], 65: [0, 2, 0], 80: [0, 1, 0], 81: [0, 1, 0], 82: [0, 0, 1], 83: [0, 1, 0],
};

class TinyStackVM {
  private stack = new Int32Array(256);
  private callStack = new Int32Array(64);
  private memory = new Int32Array(256);
  private outBuf = new Uint8Array(65536);
  private sp = 0;
  private csp = 0;
  private oLen = 0;
  private inPtr = 0;
  private pc = 0;
  private steps = 0;
  private exitCode = 0;
  private code: Uint8Array;
  private input: Uint8Array;

  constructor(code: Uint8Array, input: Uint8Array) {
    this.code = code;
    this.input = input;
  }

  public run(): { stdout: Uint8Array; exit: number } {
    const len = this.code.length;
    while (true) {
      if (this.steps >= 1_000_000) return this.trap(10);
      if (this.pc < 0 || this.pc >= len) return this.trap(9);

      const op = this.code[this.pc];
      const meta = OPS[op];
      if (!meta) return this.trap(5);

      const [imm, pop, push] = meta;
      if (this.pc + imm >= len) return this.trap(9);

      this.steps++;
      const nextPc = this.pc + 1 + imm;

      if (this.sp < pop) return this.trap(2);
      if (this.sp - pop + push > 256) return this.trap(3);

      this.pc = nextPc;
      if (!this.exec(op, nextPc)) {
        return { stdout: this.outBuf.subarray(0, this.oLen), exit: this.exitCode };
      }
    }
  }

  private trap(code: number) {
    return { stdout: this.outBuf.subarray(0, this.oLen), exit: code };
  }

  private exec(op: number, nextPc: number): boolean {
    const s = this.stack;
    const c = this.code;

    switch (op) {
      case 0x00: this.exitCode = 0; return false;
      case 0x01: s[this.sp++] = (c[nextPc - 1] << 24) >> 24; break;
      case 0x02: s[this.sp++] = c[nextPc - 4] | (c[nextPc - 3] << 8) | (c[nextPc - 2] << 16) | (c[nextPc - 1] << 24); break;
      case 0x03: this.sp--; break;
      case 0x04: s[this.sp] = s[this.sp - 1]; this.sp++; break;
      case 0x05: {
        const t = s[this.sp - 1];
        s[this.sp - 1] = s[this.sp - 2];
        s[this.sp - 2] = t;
        break;
      }
      case 0x06: s[this.sp] = s[this.sp - 2]; this.sp++; break;
      case 0x07: {
        const a = s[this.sp - 3];
        s[this.sp - 3] = s[this.sp - 2];
        s[this.sp - 2] = s[this.sp - 1];
        s[this.sp - 1] = a;
        break;
      }
      case 0x17: s[this.sp - 1] = (-s[this.sp - 1]) | 0; break;
      case 0x1b: s[this.sp - 1] = ~s[this.sp - 1]; break;
      case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16:
      case 0x18: case 0x19: case 0x1a: case 0x1c: case 0x1d: case 0x1e:
      case 0x20: case 0x21: case 0x22: {
        const r = s[--this.sp], l = s[--this.sp];
        if (op >= 0x13 && op <= 0x16 && r === 0) { this.exitCode = 4; return false; }
        let v = 0;
        switch (op) {
          case 0x10: v = (l + r) | 0; break;
          case 0x11: v = (l - r) | 0; break;
          case 0x12: v = Math.imul(l, r); break;
          case 0x13: v = l === -0x80000000 && r === -1 ? l : ((l / r) | 0); break;
          case 0x14: v = r === -1 ? 0 : ((l % r) | 0); break;
          case 0x15: v = Math.floor((l >>> 0) / (r >>> 0)) | 0; break;
          case 0x16: v = ((l >>> 0) % (r >>> 0)) | 0; break;
          case 0x18: v = l & r; break;
          case 0x19: v = l | r; break;
          case 0x1a: v = l ^ r; break;
          case 0x1c: v = l << (r & 31); break;
          case 0x1d: v = l >> (r & 31); break;
          case 0x1e: v = (l >>> (r & 31)) | 0; break;
          case 0x20: v = l === r ? 1 : 0; break;
          case 0x21: v = l < r ? 1 : 0; break;
          case 0x22: v = (l >>> 0) < (r >>> 0) ? 1 : 0; break;
        }
        s[this.sp++] = v;
        break;
      }
      case 0x30: case 0x31: case 0x32: {
        const off = ((c[nextPc - 2] | (c[nextPc - 1] << 8)) << 16) >> 16;
        let j = true;
        if (op !== 0x30) {
          const cond = s[--this.sp];
          j = op === 0x31 ? cond === 0 : cond !== 0;
        }
        if (j) this.pc = nextPc + off;
        break;
      }
      case 0x33:
        if (this.csp >= 64) { this.exitCode = 7; return false; }
        this.callStack[this.csp++] = nextPc;
        this.pc = c[nextPc - 2] | (c[nextPc - 1] << 8);
        break;
      case 0x34:
        if (this.csp <= 0) { this.exitCode = 8; return false; }
        this.pc = this.callStack[--this.csp];
        break;
      case 0x40: {
        const addr = s[this.sp - 1];
        if (addr < 0 || addr >= 256) { this.exitCode = 6; return false; }
        s[this.sp - 1] = this.memory[addr];
        break;
      }
      case 0x41: {
        const addr = s[--this.sp];
        const val = s[--this.sp];
        if (addr < 0 || addr >= 256) { this.exitCode = 6; return false; }
        this.memory[addr] = val;
        break;
      }
      case 0x50: {
        const buf = Buffer.from(String(s[--this.sp]), 'utf8');
        if (this.oLen + buf.length > 65536) { this.exitCode = 11; return false; }
        this.outBuf.set(buf, this.oLen);
        this.oLen += buf.length;
        break;
      }
      case 0x51:
        if (this.oLen + 1 > 65536) { this.exitCode = 11; return false; }
        this.outBuf[this.oLen++] = s[--this.sp] & 0xff;
        break;
      case 0x52:
        s[this.sp++] = this.inPtr < this.input.length ? this.input[this.inPtr++] : -1;
        break;
      case 0x53:
        this.exitCode = s[--this.sp] & 0xff;
        return false;
    }
    return true;
  }
}

function run(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

  for (const f of fs.readdirSync(inDir).sort()) {
    const prog = fs.readFileSync(path.join(inDir, f, 'program.bin'));
    const inp = fs.readFileSync(path.join(inDir, f, 'input.bin'));
    const res = new TinyStackVM(new Uint8Array(prog), new Uint8Array(inp)).run();

    const dest = path.join(outDir, f);
    fs.mkdirSync(dest, { recursive: true });
    fs.writeFileSync(path.join(dest, 'stdout.bin'), res.stdout);
    fs.writeFileSync(path.join(dest, 'exit.txt'), `${res.exit}\n`);
  }
}

run();
