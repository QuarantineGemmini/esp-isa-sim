if(xpr64)
  WRITE_SRD(sreg_t(SRS1) >> VSHAMT);
else
{
  if(SHAMT & 0x20)
    throw trap_illegal_instruction();
  WRITE_SRD(sext32(int32_t(SRS1) >> VSHAMT));
}
