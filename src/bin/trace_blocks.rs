use std::env;
use std::fs;
use std::process;

const ZSTD_MAGIC: u32 = 0xFD2FB528;
const SKIPPABLE_MAGIC_MASK: u32 = 0xFFFFFFF0;
const SKIPPABLE_MAGIC_BASE: u32 = 0x184D2A50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Block {
    index: usize,
    header_offset: usize,
    payload_offset: usize,
    payload_size: usize,
    block_type: u8,
    last: bool,
}

fn read_u32_le(buf: &[u8], pos: usize) -> Option<u32> {
    let bytes = buf.get(pos..pos + 4)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u24_le(buf: &[u8], pos: usize) -> Option<u32> {
    let bytes = buf.get(pos..pos + 3)?;
    Some((bytes[0] as u32) | ((bytes[1] as u32) << 8) | ((bytes[2] as u32) << 16))
}

fn frame_header_size(buf: &[u8], pos: usize) -> Result<usize, String> {
    let descriptor = *buf
        .get(pos + 4)
        .ok_or_else(|| format!("truncated frame descriptor at {pos}"))?;
    let fcs_flag = descriptor >> 6;
    let single_segment = (descriptor & 0x20) != 0;
    let dict_id_flag = descriptor & 0x03;

    let window_descriptor_size = if single_segment { 0 } else { 1 };
    let dict_id_size = match dict_id_flag {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => 4,
    };
    let fcs_size = match fcs_flag {
        0 if single_segment => 1,
        0 => 0,
        1 => 2,
        2 => 4,
        _ => 8,
    };
    let size = 4 + 1 + window_descriptor_size + dict_id_size + fcs_size;
    if pos + size > buf.len() {
        return Err(format!(
            "truncated frame header at {pos}: need {size} bytes"
        ));
    }
    Ok(size)
}

fn parse_blocks(buf: &[u8]) -> Result<Vec<Block>, String> {
    let mut pos = 0usize;
    let mut blocks = Vec::new();

    while pos < buf.len() {
        let magic = read_u32_le(buf, pos).ok_or_else(|| format!("truncated magic at {pos}"))?;
        if magic & SKIPPABLE_MAGIC_MASK == SKIPPABLE_MAGIC_BASE {
            let size = read_u32_le(buf, pos + 4)
                .ok_or_else(|| format!("truncated skippable size at {pos}"))?
                as usize;
            pos = pos
                .checked_add(8)
                .and_then(|v| v.checked_add(size))
                .ok_or_else(|| format!("skippable frame size overflow at {pos}"))?;
            if pos > buf.len() {
                return Err("truncated skippable frame payload".to_string());
            }
            continue;
        }
        if magic != ZSTD_MAGIC {
            return Err(format!("bad magic 0x{magic:08x} at offset {pos}"));
        }

        pos += frame_header_size(buf, pos)?;
        loop {
            let header_offset = pos;
            let header =
                read_u24_le(buf, pos).ok_or_else(|| format!("truncated block header at {pos}"))?;
            pos += 3;
            let last = (header & 1) != 0;
            let block_type = ((header >> 1) & 0x3) as u8;
            let block_size = (header >> 3) as usize;
            let payload_size = if block_type == 1 { 1 } else { block_size };
            let payload_offset = pos;
            pos = pos
                .checked_add(payload_size)
                .ok_or_else(|| format!("block payload size overflow at {header_offset}"))?;
            if pos > buf.len() {
                return Err(format!(
                    "truncated block payload at {payload_offset}: need {payload_size} bytes"
                ));
            }
            blocks.push(Block {
                index: blocks.len(),
                header_offset,
                payload_offset,
                payload_size,
                block_type,
                last,
            });
            if last {
                break;
            }
        }
    }

    Ok(blocks)
}

fn first_payload_diff(left: &[u8], right: &[u8], a: Block, b: Block) -> Option<usize> {
    let n = a.payload_size.min(b.payload_size);
    for i in 0..n {
        if left[a.payload_offset + i] != right[b.payload_offset + i] {
            return Some(i);
        }
    }
    (a.payload_size != b.payload_size).then_some(n)
}

fn usage() -> ! {
    eprintln!("usage: trace_blocks <left.zst> <right.zst>");
    process::exit(2);
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(left_path) = args.next() else {
        usage();
    };
    let Some(right_path) = args.next() else {
        usage();
    };
    if args.next().is_some() {
        usage();
    }

    let left = fs::read(&left_path).unwrap_or_else(|e| {
        eprintln!("read {left_path}: {e}");
        process::exit(1);
    });
    let right = fs::read(&right_path).unwrap_or_else(|e| {
        eprintln!("read {right_path}: {e}");
        process::exit(1);
    });

    let left_blocks = parse_blocks(&left).unwrap_or_else(|e| {
        eprintln!("{left_path}: {e}");
        process::exit(1);
    });
    let right_blocks = parse_blocks(&right).unwrap_or_else(|e| {
        eprintln!("{right_path}: {e}");
        process::exit(1);
    });

    println!(
        "left_blocks={} right_blocks={}",
        left_blocks.len(),
        right_blocks.len()
    );

    let common = left_blocks.len().min(right_blocks.len());
    for i in 0..common {
        let a = left_blocks[i];
        let b = right_blocks[i];
        if a.block_type != b.block_type
            || a.last != b.last
            || a.payload_size != b.payload_size
            || first_payload_diff(&left, &right, a, b).is_some()
        {
            println!(
                "first_different_block={} left={{header:{}, payload:{}, type:{}, last:{}, size:{}}} right={{header:{}, payload:{}, type:{}, last:{}, size:{}}}",
                i,
                a.header_offset,
                a.payload_offset,
                a.block_type,
                a.last,
                a.payload_size,
                b.header_offset,
                b.payload_offset,
                b.block_type,
                b.last,
                b.payload_size
            );
            if let Some(delta) = first_payload_diff(&left, &right, a, b) {
                println!("first_payload_delta={delta}");
            }
            return;
        }
    }

    if left_blocks.len() != right_blocks.len() {
        println!("block_count_diff_at={common}");
    } else {
        println!("blocks_identical");
    }
}
