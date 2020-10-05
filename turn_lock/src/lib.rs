use std::sync::atomic::{spin_loop_hint, AtomicUsize, Ordering};
use std::sync::Arc;

pub struct TurnLock {
    turn: Arc<AtomicUsize>,
    index: usize,
    num_turns: usize,
}

impl TurnLock {
    pub fn new(num_turns: usize) -> Vec<Self> {
        let turn = Arc::new(AtomicUsize::new(0));
        let mut r = Vec::new();
        for i in 0..num_turns {
            r.push(TurnLock {
                turn: turn.clone(),
                index: i,
                num_turns,
            })
        }
        r
    }
    pub fn wait(&mut self) {
        while self.turn.load(Ordering::Acquire) != self.index {
            spin_loop_hint();
        }
        assert_eq!(self.turn.load(Ordering::Relaxed), self.index);
    }
    pub fn next(&mut self) {
        assert_eq!(self.turn.load(Ordering::Relaxed), self.index);
        let r = self.turn.compare_and_swap(
            self.index,
            (self.index + 1) % self.num_turns,
            Ordering::Release,
        );
        if r != self.index {
            panic!("Released lock out of turn");
        }
    }

    pub fn current(&self) -> usize {
        self.turn.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use crate::TurnLock;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn three_turns() {
        let mut v = TurnLock::new(3);
        v[0].wait();
        v[0].next();
        v[1].wait();
        v[1].next();
        v[2].wait();
        v[2].next();
        v[0].wait();
        v[0].next();
        assert_eq!(v[2].current(), 1);
    }
}
