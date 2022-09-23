Dendrobates Tinctorius Azureus - The blue poison frog
======================================================


This is a minimal kernel (written in rust) meant to help idissect (reverse engineer) Intel CPUs
(hence the blue color of the chosen frog)


Everything remains to be done.


Design decision :
- Will only ever have one user process
- Should limit use of interrupts as much as possible
- Should support as many instructions as possible, make sure to properly enable
  all floating points and vector extensions


- [ ] Get a kernel to boot
- [ ] Get serial console
- [ ] Deal with cpuid / floating point niceties
- [ ] Deal with the user mode switch

Known good rust version : 1.57.0-nightly (9a28ac83c 2021-09-18)
