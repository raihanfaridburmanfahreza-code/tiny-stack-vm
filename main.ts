import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const STACK_MAX = 256, CALLS_MAX = 64, MEMORY_WORDS = 256, STEP_LIMIT = 1_000_000, OUTPUT_MAX = 65_536;
const TRAP = { UNDERFLOW: 2, OVERFLOW: 3, DIV_ZERO: 4, BAD_OPCODE: 5, MEMORY: 6, CALL_DEPTH: 7, RET_EMPTY: 8, PC: 9, STEPS: 10, OUTPUT: 11 } as const;

type Op = [name: string, imm: number, pops: number, pushes: number];
const OPS: [number, string, number, number, number][] = [
  [0x00, 'HALT', 0, 0, 0], [0x01, 'PUSH8', 1, 0, 1], [0x02, 'PUSH32', 4, 0, 1], [0x03, 'POP', 0, 1, 0],
  [0x04, 'DUP', 0, 1, 2], [0x05, 'SWAP', 0, 2, 2], [0x06, 'OVER', 0, 2, 3], [0x07, 'ROT', 0, 3, 3],
  [0x10, 'ADD', 0, 2, 1], [0x11, 'SUB', 0, 2, 1], [0x12, 'MUL', 0, 2, 1], [0x13, 'DIV', 0, 2, 1],
  [0x14, 'MOD', 0, 2, 1], [0x15, 'DIVU', 0, 2, 1], [0x16, 'MODU', 0, 2, 1], [0x17, 'NEG', 0, 1, 1],
  [0x18, 'AND', 0, 2, 1], [0x19, 'OR', 0, 2, 1], [0x1a, 'XOR', 0, 2, 1], [0x1b, 'NOT', 0, 1, 1],
  [0x1c, 'SHL', 0, 2, 1], [0x1d, 'SHR', 0, 2, 1], [0x1e, 'SHRU', 0, 2, 1], [0x20, 'EQ', 0, 2, 1],
  [0x21, 'LT', 0, 2, 1], [0x22, 'LTU', 0, 2, 1], [0x30, 'JMP', 2, 0, 0], [0x31, 'JZ', 2, 1, 0],
  [0x32, 'JNZ', 2, 1, 0], [0x33, 'CALL', 2, 0, 0], [0x34, 'RET', 0, 0, 0], [0x40, 'LOAD', 0, 1, 1],
  [0x41, 'STORE', 0, 2, 0], [0x50, 'PRINT', 0, 1, 0], [0x51, 'PUTC', 0, 1, 0], [0x52, 'IN', 0, 0, 1],
  [0x53, 'EXIT', 0, 1, 0],
];

const OP_TABLE: Array<Op | undefined> = new Array(256);
for (const [code, name, imm, pops, pushes] of OPS) OP_TABLE[code] = [name, imm, pops, pushes];
const enc = new TextEncoder();

function run(p: Uint8Array, input: Uint8Array): { stdout: Uint8Array; exit: number } {
  const n = p.length, D = new Int32Array(STACK_MAX), C = new Int32Array(CALLS_MAX), M = new Int32Array(MEMORY_WORDS);
  const out = new Uint8Array(OUTPUT_MAX);
  let sp = 0, csp = 0, olen = 0, inPos = 0, pc = 0, steps = 0;
  const done = (exit: number) => ({ stdout: out.subarray(0, olen), exit });
  const trap = (code: number) => done(code);

  for (;;) {
    if (steps === STEP_LIMIT) return trap(TRAP.STEPS);
    if (pc < 0 || pc >= n) return trap(TRAP.PC);
    const op = OP_TABLE[p[pc]];
    if (!op) return trap(TRAP.BAD_OPCODE);
    const [name, imm, pops, pushes] = op;
    if (pc + imm >= n) return trap(TRAP.PC);
    steps++;
    const next = pc + 1 + imm;
    if (sp < pops) return trap(TRAP.UNDERFLOW);
    if (sp - pops + pushes > STACK_MAX) return trap(TRAP.OVERFLOW);
    pc = next;

    switch (name) {
      case 'HALT': return done(0);
      case 'PUSH8': D[sp++] = (p[next - 1] << 24) >> 24; break;
      case 'PUSH32': D[sp++] = p[next - 4] | (p[next - 3] << 8) | (p[next - 2] << 16) | (p[next - 1] << 24); break;
      case 'POP': sp--; break;
      case 'DUP': D[sp] = D[sp - 1]; sp++; break;
      case 'SWAP': { const t = D[sp - 1]; D[sp - 1] = D[sp - 2]; D[sp - 2] = t; break; }
      case 'OVER': D[sp] = D[sp - 2]; sp++; break;
      case 'ROT': { const a = D[sp - 3]; D[sp - 3] = D[sp - 2]; D[sp - 2] = D[sp - 1]; D[sp - 1] = a; break; }
      case 'NEG': D[sp - 1] = -D[sp - 1] | 0; break;
      case 'NOT': D[sp - 1] = ~D[sp - 1]; break;
      case 'ADD': case 'SUB': case 'MUL': case 'DIV': case 'MOD': case 'DIVU': case 'MODU':
      case 'AND': case 'OR': case 'XOR': case 'SHL': case 'SHR': case 'SHRU': case 'EQ': case 'LT': case 'LTU': {
        const b = D[sp - 1], a = D[sp - 2];
        let r = 0;
        switch (name) {
          case 'ADD': r = (a + b) | 0; break;
          case 'SUB': r = (a - b) | 0; break;
          case 'MUL': r = Math.imul(a, b); break;
          case 'DIV': if (b === 0) return trap(TRAP.DIV_ZERO); r = a === -2147483648 && b === -1 ? a : (a / b) | 0; break;
          case 'MOD': if (b === 0) return trap(TRAP.DIV_ZERO); r = b === -1 ? 0 : (a % b) | 0; break;
          case 'DIVU': if (b === 0) return trap(TRAP.DIV_ZERO); r = Math.floor((a >>> 0) / (b >>> 0)) | 0; break;
          case 'MODU': if (b === 0) return trap(TRAP.DIV_ZERO); r = (a >>> 0) % (b >>> 0) | 0; break;
          case 'AND': r = a & b; break;
          case 'OR': r = a | b; break;
          case 'XOR': r = a ^ b; break;
          case 'SHL': r = a << (b & 31); break;
          case 'SHR': r = a >> (b & 31); break;
          case 'SHRU': r = (a >>> (b & 31)) | 0; break;
          case 'EQ': r = a === b ? 1 : 0; break;
          case 'LT': r = a < b ? 1 : 0; break;
          case 'LTU': r = (a >>> 0) < (b >>> 0) ? 1 : 0; break;
        }
        sp -= 2;
        D[sp++] = r;
        break;
      }
      case 'JMP': case 'JZ': case 'JNZ': {
        const rel = ((p[next - 2] | (p[next - 1] << 8)) << 16) >> 16;
        let take = true;
        if (name !== 'JMP') {
          const v = D[--sp];
          take = name === 'JZ' ? v === 0 : v !== 0;
        }
        if (take) pc = next + rel;
        break;
      }
      case 'CALL':
        if (csp === CALLS_MAX) return trap(TRAP.CALL_DEPTH);
        C[csp++] = next;
        pc = p[next - 2] | (p[next - 1] << 8);
        break;
      case 'RET':
        if (csp === 0) return trap(TRAP.RET_EMPTY);
        pc = C[--csp];
        break;
      case 'LOAD': {
        const addr = D[sp - 1];
        if (addr < 0 || addr >= MEMORY_WORDS) return trap(TRAP.MEMORY);
        D[sp - 1] = M[addr];
        break;
      }
      case 'STORE': {
        const addr = D[sp - 1], v = D[sp - 2];
        if (addr < 0 || addr >= MEMORY_WORDS) return trap(TRAP.MEMORY);
        sp -= 2;
        M[addr] = v;
        break;
      }
      case 'PRINT': {
        const bytes = enc.encode(String(D[sp - 1]));
        if (olen + bytes.length > OUTPUT_MAX) return trap(TRAP.OUTPUT);
        sp--;
        out.set(bytes, olen);
        olen += bytes.length;
        break;
      }
      case 'PUTC':
        if (olen + 1 > OUTPUT_MAX) return trap(TRAP.OUTPUT);
        out[olen++] = D[--sp] & 255;
        break;
      case 'IN':
        D[sp++] = inPos < input.length ? input[inPos++] : -1;
        break;
      case 'EXIT':
        return done(D[--sp] & 255);
    }
  }
}

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

for (const id of readdirSync(I).sort()) {
  const prog = new Uint8Array(readFileSync(join(I, id, 'program.bin')));
  const inp = new Uint8Array(readFileSync(join(I, id, 'input.bin')));
  const r = run(prog, inp);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'stdout.bin'), r.stdout);
  writeFileSync(join(O, id, 'exit.txt'), `${r.exit}\n`);
}
