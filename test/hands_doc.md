# DexterousFingerParameter

## Class Information
- **Class Name:** DexterousFingerParameter
- **Effective Version:** >= v1.0.6.0

## Description
This class definition represents a dexterous finger parameter. Different dexterous hands have different motion parameters.

The following parameters all represent a scaling factor. When using them, you need to convert them into the corresponding coefficients based on the specifications of the dexterous hand you are using.

## Fields

### seq
**Range:** 0~5, represents the sequence of the fingers, as follows:
- 0: Little finger
- 1: Ring finger
- 2: Middle finger
- 3: Index finger
- 4: Thumb bending
- 5: Thumb rotation

### angle
**Range:** 0~1000, 0 represents the fully closed state, 1000 represents the fully open state.

Different fingers have different ranges:
- The range 0~1000 of the thumb rotation corresponds to 90-165°
- The range 0~1000 of the thumb bending corresponds to -130~53.6°
- The range 0~1000 of the other fingers corresponds to 19°-176.7°

### force
**Description:** Represents the finger's force value. Different fingers have different ranges. The finger will stop moving when it receives the corresponding feedback force.

**Force ranges:**
- The range 0~1500 of the thumb rotation and bending corresponds to 0~1.5kg
- The range 0~1000 of the other fingers corresponds to 0~1kg

### speed
**Range:** 0~1000

**Speed unit:** Not specified. The Inspire RH56 dexterous hand does not provide a unit, range, or dimension for speed. The speed can only be adjusted using values from 0 to 1000, which do not correspond to an absolute value.