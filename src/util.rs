use std::io;

/// Decode SPIR-V from bytes.
///
/// This function handles SPIR-V of arbitrary endianness gracefully, and returns correctly aligned
/// storage.
/// 
/// Copied from `ash`.
pub fn read_spv<R: io::Read + io::Seek>(x: &mut R) -> io::Result<Vec<u32>> {
    // TODO use stream_len() once it is stabilized and remove the subsequent rewind() call
    let size = x.seek(io::SeekFrom::End(0))?;
    x.rewind()?;
    if size % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input length not divisible by 4",
        ));
    }
    if size > usize::MAX as u64 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "input too long"));
    }
    let words = (size / 4) as usize;
    // https://github.com/ash-rs/ash/issues/354:
    // Zero-initialize the result to prevent read_exact from possibly
    // reading uninitialized memory.
    let mut result = vec![0u32; words];
    x.read_exact(unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr().cast::<u8>(), words * 4)
    })?;
    const MAGIC_NUMBER: u32 = 0x0723_0203;
    if !result.is_empty() && result[0] == MAGIC_NUMBER.swap_bytes() {
        for word in &mut result {
            *word = word.swap_bytes();
        }
    }
    if result.is_empty() || result[0] != MAGIC_NUMBER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input missing SPIR-V magic number",
        ));
    }
    Ok(result)
}
