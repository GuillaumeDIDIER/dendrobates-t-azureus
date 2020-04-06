pub struct AddressHasher<'a> {
    masks: &'a [usize],
}

fn hash(addr: usize, mask: usize) -> u32 {
    (addr & mask).count_ones() & 1
}

impl AddressHasher<'_> {
    pub fn new(masks: &[usize]) -> AddressHasher {
        AddressHasher { masks }
    }
    pub fn hash(&self, addr: usize) -> u32 {
        let mut res = 0;
        for mask in self.masks {
            res <<= 1;
            res |= hash(addr, *mask);
        }
        res
    }
}
