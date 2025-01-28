#![cfg(feature = "use_std")]
extern crate std;

use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut};
use core::ptr;
use core::ptr::NonNull;
use core::slice::{from_raw_parts, from_raw_parts_mut};
use nix::sys::mman;
use std::convert::TryFrom;

/* from linux kernel headers.
#define HUGETLB_FLAG_ENCODE_SHIFT       26
#define HUGETLB_FLAG_ENCODE_MASK        0x3f

#define HUGETLB_FLAG_ENCODE_64KB        (16 << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_512KB       (19 << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_1MB         (20 << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_2MB         (21 << HUGETLB_FLAG_ENCODE_SHIFT)
*/
/** Safety issue : if T is non triviably constructable and destructable this is dangerous */
pub struct MMappedMemory<T> {
    pointer: NonNull<T>,
    size: usize,
}

impl<T> MMappedMemory<T> {
    pub fn try_new(
        size: usize,
        huge: bool,
        executable: bool,
        initializer: impl Fn(usize) -> T,
    ) -> Result<MMappedMemory<T>, nix::Error> {
        assert_ne!(size_of::<T>(), 0);
        let prot = if executable {
            mman::ProtFlags::PROT_READ | mman::ProtFlags::PROT_WRITE | mman::ProtFlags::PROT_EXEC
        } else {
            mman::ProtFlags::PROT_READ | mman::ProtFlags::PROT_WRITE
        };
        let r = unsafe {
            mman::mmap_anonymous(
                None,
                NonZeroUsize::try_from(size * size_of::<T>()).unwrap(),
                prot,
                mman::MapFlags::MAP_PRIVATE
                    | if huge {
                        mman::MapFlags::MAP_HUGETLB | mman::MapFlags::MAP_ANONYMOUS
                    } else {
                        mman::MapFlags::MAP_ANONYMOUS
                    },
            )
        };
        match r {
            Ok(p) => {
                let s: MMappedMemory<T> = MMappedMemory {
                    pointer: p.cast(),
                    size,
                };
                for i in 0..s.size {
                    unsafe { ptr::write(s.pointer.as_ptr().add(i), initializer(i)) };
                }
                Ok(s)
            }
            Err(e) => Err(e),
        }
    }
    /*
        pub fn try_new_uninit(
            size: usize,
            huge: bool,
        ) -> Result<MMappedMemory<MaybeUninit<T>>, nix::Error> {
            assert_ne!(size_of::<T>(), 0);
            if let Some(p) = unsafe {
                let p = mman::mmap(
                    null_mut(),
                    size * size_of::<T>(),
                    mman::ProtFlags::PROT_READ | mman::ProtFlags::PROT_WRITE,
                    mman::MapFlags::MAP_PRIVATE
                        | mman::MapFlags::MAP_ANONYMOUS
                        | if huge {
                            mman::MapFlags::MAP_HUGETLB
                        } else {
                            mman::MapFlags::MAP_ANONYMOUS
                        },
                    -1,
                    0,
                )?;
                let pointer_T = p as *mut T;
                Unique::new(pointer_T)
            } {
                let mut s = MMappedMemory { pointer: p, size };
                Ok(s)
            } else {
                Err(nix::Error::Sys(EINVAL))
            }
        }
    */
    pub fn new(
        size: usize,
        huge: bool,
        executable: bool,
        init: impl Fn(usize) -> T,
    ) -> MMappedMemory<T> {
        Self::try_new(size, huge, executable, init).unwrap()
    }

    pub fn slice(&self) -> &[T] {
        unsafe { from_raw_parts(self.pointer.as_ptr(), self.size) }
    }

    pub fn slice_mut(&mut self) -> &mut [T] {
        unsafe { from_raw_parts_mut(self.pointer.as_ptr(), self.size) }
    }
}

impl<T> Drop for MMappedMemory<T> {
    fn drop(&mut self) {
        unsafe {
            mman::munmap(self.pointer.cast(), self.size).unwrap();
        }
    }
}

impl<T> Deref for MMappedMemory<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.slice()
    }
}

impl<T> DerefMut for MMappedMemory<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice_mut()
    }
}

impl<T> AsRef<[T]> for MMappedMemory<T> {
    fn as_ref(&self) -> &[T] {
        self.slice()
    }
}

impl<T> AsMut<[T]> for MMappedMemory<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.slice_mut()
    }
}

impl<T> Borrow<[T]> for MMappedMemory<T> {
    fn borrow(&self) -> &[T] {
        self.slice()
    }
}

impl<T> BorrowMut<[T]> for MMappedMemory<T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.slice_mut()
    }
}

// It owns the memory, so it should be safe to send.
unsafe impl<T> Send for MMappedMemory<T> {}
