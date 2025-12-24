// Mini speck encryption suite

function hex_to_uint32(x)
{
    return parseInt(x, 16);
}

/**
 * Strip PKCS7 padding.
 * @param p Plaintext featuring padding blocks.
 * @param bs The block size.
 * @todo This code should error if the last byte is not <= bs, as that would indicate a padding error &/or decryption error.
 * However, for compatibilty, we need to keep the no throw behaviour.
 */
function strip_pksc7_padding(p, bs)
{
    var n = 0;
    if (p.length && ((n = p[p.length - 1]) <= bs && (n > 0))) p.splice(p.length - n, n);
}

function speck_cbc_encrypt_64_128(p, c, iv_u64, k1_u64, k2_u64)
{
    let kx = this.speck_expand_key_64_128(k1_u64.concat(k2_u64));
    this.speck_cbc_encrypt_64_128_expanded(c, p, iv_u64, kx);
}

function speck_cbc_decrypt_64_128(p, c, iv_u64, k1_u64, k2_u64)
{
    let kx = this.speck_expand_key_64_128(k1_u64.concat(k2_u64));
    this.speck_cbc_decrypt_64_128_expanded(p, c, iv_u64, kx);
    this.strip_pksc7_padding(p, 8);
}

function speck_cbc_encrypt_64_128_expanded(c, p, iv, k)
{
    const block_size = 8;
    let p_copy = Array.from(p);

    let padding = this.speck_cbc_calculate_plaintext_padding(p_copy, block_size);
    this.pksc7(p_copy, padding);

    let last_block = iv;

    for (let i = 0; i < p_copy.length; i += block_size)
    {
        let block_8_bit = p_copy.slice(i, i + block_size);
        let block_32_bit = this.bytes_to_uint32_t(block_8_bit);
        this.initialize_vector(block_32_bit, last_block)
        this.speck_encrypt_64_128(block_32_bit, k);
        Array.prototype.push.apply(c, this.uint32_t_array_to_bytes(block_32_bit, true))
        last_block = block_32_bit;
    }
}

function speck_cbc_decrypt_64_128_expanded(p, c, iv, k)
{
    let last_ui64_block = iv;

    let c_ui32 = this.bytes_to_uint32_t(c);

    for (var i = 0; i < c_ui32.length;)
    {
        let c_i64 = [c_ui32[i], c_ui32[i + 1]];
        this.speck_decrypt_64_128(c_i64, k);
        this.initialize_vector(c_i64, last_ui64_block);

        Array.prototype.push.apply(p, this.uint32_t_array_to_bytes(c_i64, false))

        i += 2;
        last_ui64_block = [c_ui32[i - 2], c_ui32[i - 1]];
    }
}

/**
 * Calculate the number of padding bytes for the given block size.
 * @param p Plaintext array to calculate the padding for.
 * @param bs Block size for the cipher.
 * @return Number of padding bytes required.
 */
function speck_cbc_calculate_plaintext_padding(p, bs)
{
    let n = 0;

    for (let i = p.length; i > 0; i -= bs)
    {
        if (i < bs)
        {
            n = bs - i;
            break;
        }
    }

    if (n == 0) n = bs;

    return n;
}

function bytes_to_uint32_t(t)
{
    let rv = [];
    let buf = new Int8Array(t);
    let buf2 = new Int32Array(buf.buffer);
    rv = Array.from(buf2);
    return rv;
}

function uint32_t_array_to_bytes(i32_arr, allow_negative)
{
    let rv = [];
    let buf2 = new Int32Array(i32_arr);
    let buf = allow_negative ? new Int8Array(buf2.buffer) : new Uint8Array(buf2.buffer);
    rv = Array.from(buf);

    return rv;
}

function initialize_vector(b, v)
{
    for (var i = 0; i < b.length; i++) b[i] = b[i] ^ v[i];
}

function pksc7(p, n)
{
    let i = n;
    if (n > 0) while (i) { p.push(n); --i; };
    return p;
}

function speck_expand_key_64_128(key)
{
    var k = key[3];
    var expanded = [];

    for (var i = 0, j; i < 27; ++i)
    {
        expanded[i] = k;
        j = 2 - i % 3;
        key[j] = (key[j] << 24 | key[j] >>> 8) + k ^ i;
        k = (k << 3 | k >>> 29) ^ key[j];
    }

    return expanded;
}

RR = function (x, r, w) { return ((x >>> r) | (x << (w - r))); };
RL = function(x, r, w) { return ((x << r) | (x >>> (w - r))); };

function speck_encrypt_64_128(b, k)
{
    for (var i = 0; i < 27; i++)
    {
        b[1] = (RR(b[1], 8, 32) + b[0]) ^ k[i];
        b[0] = RL(b[0], 3, 32) ^ b[1];
    }
}

function speck_decrypt_64_128(b, k)
{
    for (var i = 27; i > 0; i--)
    {
        b[0] = b[0] ^ b[1];
        b[0] = RR(b[0], 3, 32);
        b[1] = (b[1] ^ k[i-1]) - b[0];
        b[1] = RL(b[1], 8, 32);
    }
};



// Constructs a HMAC function from the specified hash function, assumes 64 bit block size as uint32[2]
function hmac_64(hash)
{
    return function(key, message)
    {
        var k = key;
        var o_key_pad = [], i_key_pad = [];
        o_key_pad[0] = k[0] ^ (0x5c5c5c5c >>> 0);
        o_key_pad[1] = k[1] ^ (0x5c5c5c5c >>> 0);
        i_key_pad[0] = k[0] ^ (0x36363636 >>> 0);
        i_key_pad[1] = k[1] ^ (0x36363636 >>> 0);
        var m1 = uint32_t_array_to_bytes(o_key_pad, false);
        var m2 = (uint32_t_array_to_bytes(i_key_pad, false)).concat(Array.from(message));
        var m3 = hash(m2);
        return hash(m1.concat(uint32_t_array_to_bytes([m3[1], m3[0]])));
    }
}

// A 64-bit hash function based on Speck, message should be an array of uint32 (2 elements per block)
function speck_hash_64(message)
{
    var i;
    var iv = [ 0x11223344, 0x55667788 ];

    // Davis-Mayer Compression function
    function compress(h1, mb)
    {
        var k = speck_expand_key_64_128(h1.concat(h1));  // Key is 128 bit so need to double the size
        speck_encrypt_64_128(mb, k);
        var h2 = []
        h2[1] = mb[0] ^ h1[1];
        h2[0] = mb[1] ^ h1[0];
        return h2;
    }

    // Convert and pad message
    var paddedMessage = Array.from(message);
    i = 8 - (message.length % 8);
    if (i == 0) i = 8;
    while (i) { paddedMessage.push(0); --i; };
    var mb = bytes_to_uint32_t(paddedMessage);

    // Merkle–Damgård construction
    var h = iv;
    for (i=0; i<mb.length / 2; i++)
    {
        var m = [];
        m[0] = mb[i*2];
        m[1] = mb[(i*2) + 1];
        h = compress(h, m);
    }
    return h;
}


hmac_speck_64 = hmac_64(speck_hash_64);

function toHex(array)
{
    let rv;
    rv = new Int8Array(array); //uint32_t_array_to_bytes(array));
    rv = new Uint8Array(rv.reverse().buffer);
    rv = rv.reduceRight((acc, byte) => {
        let str = byte.toString(16);
        if (str.length === 1) {
            str = `0${str}`;
        }
        return acc + str;
    }, '');
    return rv;
}

/**
 * Converts a hex string (reversed byte order) back into a Uint32Array.
 * Assumes input was generated by toHex().
 * @param {string} hexStr - Hex string from `toHex()`.
 * @returns {Uint32Array}
 */
function fromHex(hexStr) {
    const bytes = [];

    // Read hex in 2-character chunks
    for (let i = 0; i < hexStr.length; i += 2) {
        bytes.push(parseInt(hexStr.substr(i, 2), 16));
    }

    // Reverse the byte order (to match toHex logic)
    bytes.reverse();

    // Convert to Int8Array for buffer reinterpretation
    const i8 = new Int8Array(bytes);
    const i32 = new Int32Array(i8.buffer);
    return new Uint32Array(i32);
}

function toHexReverse(array)
{
    let rv;
    rv = new Int8Array(uint32_t_array_to_bytes(array));
    rv = new Uint8Array(rv.buffer);
    rv = rv.reduceRight((acc, byte) => {
        let str = byte.toString(16);
        if (str.length === 1) {
            str = `0${str}`;
        }
        return acc + str;
    }, '');
    return rv;
}

function stringToBytes(str) {
    let utf8Encode = new TextEncoder();
    return utf8Encode.encode(str);
}

/**
 * Derives a key using PBKDF2 with HMAC-Speck64.
 * @param {Uint8Array} password - Password as byte array.
 * @param {Uint8Array} salt - Salt as byte array.
 * @param {number} iterations - Number of iterations (e.g., 100000).
 * @param {number} keyLen - Desired output key length in bytes (must be multiple of 8).
 * @returns {Uint8Array} Derived key.
 */
 function pbkdf2_speck64(password, salt, iterations, keyLen)
{
    const hmac = hmac_64(speck_hash_64);
    const blockSize = 8; // 64-bit blocks
    const blocks = Math.ceil(keyLen / blockSize);

    // Convert password into 64-bit key (first 8 bytes)
    let password_bytes = stringToBytes(password + password + password + password);
    const pwdBlocks = bytes_to_uint32_t(password_bytes.slice(0, 8));

    const derived = [];

    for (let blockIndex = 1; blockIndex <= blocks; blockIndex++) {
        // Salt || blockIndex (big endian 4-byte)
        const blockBytes = new Uint8Array(4);
        new DataView(blockBytes.buffer).setUint32(0, blockIndex, false);
        const msg = new Uint8Array([...salt, ...blockBytes]);

        let u = hmac(pwdBlocks, msg);  // U1
        let t = [...u];                // T = U1

        for (let i = 1; i < iterations; i++) {
            u = hmac(pwdBlocks, uint32_t_array_to_bytes(u, false));
            t[0] ^= u[0];
            t[1] ^= u[1];
        }

        derived.push(...uint32_t_array_to_bytes(t, false));
    }

    return new Uint8Array(derived.slice(0, keyLen));
}

