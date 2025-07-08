const assert = require('assert');
const fsPromises = require('node:fs/promises');
const sharp = require('sharp');
const llcomp_ext = '.llcomp';
class RangeEncoder {
    constructor(put_byte) {
        this.outstanding_count = 0;
        this.outstanding_byte = -1;
        this.low = 0;
        this.range = 0xFF00;
        this.put_byte = put_byte;
    }

    renorm_encoder() {
        while (this.range < 0x100) {
            if (this.outstanding_byte < 0) {
                this.outstanding_byte = this.low >>> 8;
            }
            else if (this.low <= 0xFF00) {
                this.put_byte(this.outstanding_byte);
                for (; this.outstanding_count; this.outstanding_count--)
                    this.put_byte(0xFF);
                this.outstanding_byte = this.low >>> 8;
            }
            else if (this.low >= 0x10000) {
                this.put_byte(this.outstanding_byte + 1);
                for (; this.outstanding_count; this.outstanding_count--)
                    this.put_byte(0x00);
                this.outstanding_byte = (this.low >> 8) & 0xFF;
            }
            else {
                this.outstanding_count++;
            }
            this.low = (this.low & 0xFF) << 8;
            this.range <<= 8;
        }
    }

    put(bit, probability) {
        const range1 = Math.max(1, Math.min(this.range - 1, this.range * probability | 0));
        assert(range1 < this.range);
        assert(range1 > 0);
        if (!bit) {
            this.range -= range1;
        }
        else {
            this.low += this.range - range1;
            this.range = range1;
        }
        assert(this.range >= 1);
        this.renorm_encoder();
    }

    finish() {
        this.range = 0xFF;
        this.low += 0xFF;
        this.renorm_encoder();
        this.range = 0xFF;
        this.renorm_encoder();
    }
}

class RangeDecoder {
    constructor(get_byte) {
        this.range = 0xFF00;
        this.get_byte = get_byte;
        this.low = this.get_byte() << 8;
        this.low |= this.get_byte();
    }

    refill() {
        if (this.range < 0x100) {
            this.range <<= 8;
            this.low <<= 8;
            this.low += this.get_byte();
        }
    }

    get(probability) {
        const range1 = Math.max(1, Math.min(this.range - 1, this.range * probability | 0));
        this.range -= range1;
        if (this.low < this.range) {
            this.refill();
            return 0;
        }
        else {
            this.low -= this.range;
            this.range = range1;
            this.refill();
            return 1;
        }
    }
}

const NEXT_STATE_MPS = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
    119, 120, 121, 122, 123, 124, 125, 124, 125, 126, 127
];

const NEXT_STATE_LPS = [
    1, 0, 0, 1, 2, 3, 4, 5, 4, 5, 8, 9, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 18, 19, 22,
    23, 22, 23, 24, 25, 26, 27, 26, 27, 30, 31, 30, 31, 32, 33, 32, 33, 36, 37, 36, 37, 38, 39, 38,
    39, 42, 43, 42, 43, 44, 45, 44, 45, 46, 47, 48, 49, 48, 49, 50, 51, 52, 53, 52, 53, 54, 55, 54,
    55, 56, 57, 58, 59, 58, 59, 60, 61, 60, 61, 60, 61, 62, 63, 64, 65, 64, 65, 66, 67, 66, 67, 66,
    67, 68, 69, 68, 69, 70, 71, 70, 71, 70, 71, 72, 73, 72, 73, 72, 73, 74, 75, 74, 75, 74, 75, 76,
    77, 76, 77, 126, 127
];

const MPS_PROBABILITY = [
    0.5156,0.5405,0.5615,0.5825,0.6016,0.6207,0.6398,0.6570,
    0.6723,0.6875,0.7028,0.7162,0.7295,0.7410,0.7525,0.7639,
    0.7754,0.7849,0.7945,0.8040,0.8117,0.8212,0.8289,0.8365,
    0.8422,0.8499,0.8556,0.8613,0.8671,0.8728,0.8785,0.8823,
    0.8881,0.8919,0.8957,0.8995,0.9033,0.9072,0.9110,0.9148,
    0.9167,0.9205,0.9224,0.9263,0.9282,0.9301,0.9320,0.9339,
    0.9358,0.9377,0.9396,0.9415,0.9434,0.9454,0.9473,0.9473,
    0.9492,0.9511,0.9511,0.9530,0.9530,0.9549,0.9568,0.9702
];

class Model3 {
    constructor(contexts_count = (11 * 11 * 11 * 5 * 5 + 1) / 2, states_count = 8) {
        this.contexts_count = contexts_count;
        this.states_count = states_count;
        this.states = new Uint8Array(this.states_count * this.contexts_count);
        this.states.fill(0);
    }
    reset() {
        this.states.fill(0);
    }
    P(context, bitpos) {
        const index = Math.min(this.contexts_count - 1, context) * this.states_count + Math.min(this.states_count - 1, bitpos);
        const state = this.states[index];
        const mps = state & 1;
        const p_mps = MPS_PROBABILITY[state >>> 1];
        return (mps) ? p_mps : 1 - p_mps;
    }

    update(context, bitpos, bit) {
        assert(bit === 0 || bit === 1);
        const index = Math.min(this.contexts_count - 1, context) * this.states_count + Math.min(this.states_count - 1, bitpos);
        const state = this.states[index];
        this.states[index] = (state & 1) === bit ? NEXT_STATE_MPS[state] : NEXT_STATE_LPS[state];
    }
}

function putSymbol(v, isSigned, putRac) {
    let i;

    if (putRac == null)
        return;
    if (v) {
        const a = Math.abs(v) | 0;
        const e = Math.floor(Math.log2(a)) | 0;

        putRac(0, 0);

        let ctx = 1;
        for (i = 0; i < e; i++) {
            putRac(Math.min(ctx++, 4), 1);
        }
        putRac(Math.min(ctx++, 4), 0);

        ctx = 5;
        for (i = e - 1; i >= 0; i--) {
            putRac(Math.min(ctx++, 6), (a >>> i) & 1);
        }

        if (isSigned) {
            putRac(7, v < 0 ? 1 : 0);
        }
    } else {
        putRac(0, 1);
    }
}

function getSymbol(isSigned, getRac) {
    if (getRac == null) return 0;

    if (getRac(0)) return 0;

    let e = 0;
    let ctx = 1;
    while (getRac(Math.min(ctx++, 4))) {
        e++;
        if (e > 31) {
            throw new Error("Invalid exponent");
        }
    }

    let value = 1;
    ctx = 5;
    for (let i = e - 1; i >= 0; i--) {
        value += value + getRac(Math.min(ctx++, 6));
    }

    if (isSigned) {
        if (getRac(7))
            value = -value;
    }
    return value;
}

const quant5_table = [
    0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1,
];

const quant11_table = [
    0, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
    -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
    -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
    -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
    -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
    -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -4, -4,
    -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
    -4, -4, -4, -4, -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -1,
];

function quant11(x) {
    return quant11_table[Math.max(-128, Math.min(127, x)) & 0xFF];
}
function quant5(x) {
    return quant5_table[Math.max(-128, Math.min(127, x)) & 0xFF];
}

function median(a, b, c) {
    if (a > b) {
        if (c > b) {
            if (c > a) b = a;
            else b = c;
        }
    } else {
        if (b > c) {
            if (c > a) b = c;
            else b = a;
        }
    }
    return b;
}

class CPromise {
    constructor() {
        this.promise = new Promise((resolve, reject) => {
            this.resolve = resolve;
            this.reject = reject;
        });
    }
    async wait() {
        return this.promise;
    }
    resolve(value) {
        this.resolve(value);
    }
    reject(error) {
        this.reject(error);
    }
}

/* it's a hack to yield to the event loop */
const Yield = async ()=>{
    const _= new CPromise();
    setImmediate(_.resolve);
    return _.wait();
}

async function compressImage(raw,_yield=Yield, _yield_period=256) {
    if (typeof _yield !== 'function') _yield_period=void 0;
    let width = raw.info.width;
    let height = raw.info.height;
    let channels = raw.info.channels;
    let size = width * height * channels;
    let stride = width * channels;
    assert(channels == 3 || channels == 4);
    let has_alpha = channels > 3 || channels == 2;
    assert(size == raw.data.length);
    let rgb = raw.data;
    let buffer = Buffer.alloc(size);
    let write_pos = 0;
    let read_pos = 0;

    function writeU8(x) {
        buffer.writeUint8(x, write_pos); write_pos += 1;
    }
    function writeU16(x) {
        buffer.writeUint16LE(x, write_pos); write_pos += 2;
    }

    writeU8(0x77);
    writeU8(channels);
    writeU16(width);
    writeU16(height);

    let comp = new RangeEncoder(x => {
        writeU8(x);
    });

    const lines = [new Int16Array(stride), new Int16Array(stride), new Int16Array(stride)];
    const model = new Model3();
    let pos = 0;
    const x1 = channels;
    const x2 = channels * 2;
    let Y_period = 0;
    for (let h = 0; h < height; ++h) {
        const line0 = lines[h % 3];
        const line1 = lines[(h + 3 - 1) % 3];
        const line2 = lines[(h + 3 - 2) % 3];
        for (let w = 0; w < width; ++w) {
            if (Y_period++ === _yield_period) {
                Y_period = 0;
                await _yield();
            }
            const x = w * channels;
            if (channels >= 3) {
                let r = rgb[pos];
                let g = rgb[pos + 1];
                let b = rgb[pos + 2];
                b -= g;
                r -= g;
                g += (b + r) / 4 | 0;

                line0[x + 0] = r;
                line0[x + 1] = g;
                line0[x + 2] = b;
                for (let i = 3; i < channels; i++) {
                    line0[x + i] = rgb[pos + i];
                }
            } else {
                for (let i = 0; i < channels; i++) {
                    line0[x + i] = rgb[pos + i];
                }
            }
            pos += channels;
            for (let i = 0; i < channels; i++) {
                const l = w > 0 ? line0[x - x1 + i] : h > 0 ? line1[x + i] : 128;
                const t = h > 0 ? line1[x + i] : l;
                const L = w > 1 ? line0[x - x2 + i] : l;
                const tl = h > 0 && w > 0 ? line1[x - x1 + i] : t;
                const tr = h > 0 && w < (width - 1) ? line1[x + x1 + i] : t;
                const T = h > 1 ? line2[x + i] : t;

                let hash = (quant11(l - tl) +
                    quant11(tl - t) * (11) +
                    quant11(t - tr) * (11 * 11));
                if (model.contexts_count > 666) {
                    hash += quant5(L - l) * (5 * 11 * 11) + quant5(T - t) * (5 * 5 * 11 * 11);
                }
                const predict = median(l, l + t - tl, t);
                let diff = (line0[x + i] - predict);

                if (hash < 0) {
                    hash = -hash;
                    diff = -diff;
                }
                assert(hash >= 0);

                putSymbol(diff, true, (bitpos, bit) => {
                    const prob = model.P(hash, bitpos);
                    comp.put(bit, prob);
                    model.update(hash, bitpos, bit);
                });
            }
        }
    }
    comp.finish();
    return buffer.subarray(0, write_pos);
}


async function decompressImage(data, _yield=Yield, _yield_period=256) {
    if (typeof _yield !== "function") _yield_period = void 0;
    const magic = data.readUint8(0); data = data.subarray(1);
    assert((magic & 0xFF) === 0x77, "Invalid magic number");
    let pos = 0;
    const channels = data.readUint8(pos); pos += 1;
    const width = data.readUInt16LE(pos); pos += 2;
    const height = data.readUInt16LE(pos); pos += 2;

    const stride = width * channels;
    let pixels = Buffer.alloc(width * height * channels);
    let rgb_pos = 0;

    const decomp = new RangeDecoder(() => {
        if (pos >= data.length)
            return 0;
        return data.readUint8(pos++);
    });
    const model = new Model3();

    const x1 = channels;
    const x2 = channels * 2;
    let Y_period = 0;
    const lines = [
        new Int16Array(stride),
        new Int16Array(stride),
        new Int16Array(stride)
    ];

    for (let h = 0; h < height; h++) {
        const line0 = lines[h % 3];
        const line1 = lines[(h + 2) % 3];
        const line2 = lines[(h + 1) % 3];

        for (let w = 0; w < width; w++) {
            if (Y_period++ === _yield_period) {
                Y_period = 0;
                await _yield();
            }
            const x = w * channels;
            for (let i = 0; i < channels; i++) {
                const l = w > 0 ? line0[x - x1 + i] : (h > 0 ? line1[x + i] : 128);
                const t = h > 0 ? line1[x + i] : l;
                const L = w > 1 ? line0[x - x2 + i] : l;
                const tl = h > 0 && w > 0 ? line1[x - x1 + i] : t;
                const tr = h > 0 && w < width - 1 ? line1[x + x1 + i] : t;
                const T = h > 1 ? line2[x + i] : t;

                let hash = (quant11(l - tl) +
                    quant11(tl - t) * 11 +
                    quant11(t - tr) * 11 * 11);

                if (model.contexts_count > 666) {
                    hash += quant5(L - l) * (5 * 11 * 11) + quant5(T - t) * (5 * 5 * 11 * 11);
                }

                const predict = median(l, l + t - tl, t);

                let neg_diff = false;
                if (hash < 0) {
                    hash = -hash;
                    neg_diff = true;
                }

                let diff = getSymbol(true, (bitpos) => {
                    const prob = model.P(hash, bitpos);
                    let bit = decomp.get(prob);
                    model.update(hash, bitpos, bit);
                    return bit;
                });

                if (neg_diff) {
                    diff = -diff;
                }
                line0[x + i] = predict + diff;
            }

            let r = line0[x + 0];
            let g = line0[x + 1];
            let b = line0[x + 2];
            g -= ((r + b) / 4 | 0);
            r += g;
            b += g;
            pixels[rgb_pos++] = Math.max(0, Math.min(255, r));
            pixels[rgb_pos++] = Math.max(0, Math.min(255, g));
            pixels[rgb_pos++] = Math.max(0, Math.min(255, b));
            for (let i = 3; i < channels; i++) {
                pixels[rgb_pos++] = line0[x + i];
            }
        }
    }
    return { data: pixels, info: { width, height, channels } };
}

async function main() {
    const fileName = process.argv[2];
    if (!fileName) {
        process.stderr.write('Usage: node range.js <filename>');
        process.exitCode = 1;
        return;
    }
    let p = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    let i = 0;

    const progress = setInterval(() => {
            process.stdout.write(p[i % p.length] + '\b');
        i += 1;
    }, 100);

    try {
        if (fileName.endsWith(llcomp_ext)) {

            process.stdout.write('Decompression...');
            const  raw = await decompressImage(await fsPromises.readFile(fileName));
            process.stdout.write('\nWriting image...');
            await sharp(raw.data, { raw: raw.info }).png().toFile(fileName + '.png');

        } else {
            process.stdout.write('Reading image...');
            const image = await sharp(fileName).raw().toBuffer({ resolveWithObject: true });
            process.stdout.write('\nCompression...');
            const compressed = await compressImage(image);
            process.stdout.write('\nWrite file...');
            await fsPromises.writeFile(fileName + llcomp_ext,compressed );
        }
        process.stdout.write(' \nComplete!\n');

    } catch (error) {
        console.error('Error:', error);
        process.stdout.write(`\bError: ${error}\n`);
    }
    finally {
        clearInterval(progress);
    }
}

main();
