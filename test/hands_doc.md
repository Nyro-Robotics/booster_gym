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

---

# ROS Interface - Hand Data Subscription

## Topic Information
- **Topic Name:** `rt/booster_hand_data`
- **Access Method:** subscribe
- **Description:** Subscribe to hand data through low level service interface

## Interface Structures

### HandReplyParam
```cpp
struct HandReplyParam {
    int32 angle;   // Refer to DexterousFingerParameter in high level api
    int32 force;   // Refer to DexterousFingerParameter in high level api  
    int32 current; // Joint temperature [0-1000] mA
    int32 error;   // See error codes below
    int32 status;  // See status codes below
    int32 temp;    // Joint temperature [0-100] °C
    int32 seq;     // Reference high level DexterousFingerParameter
};
```

### HandReplyData
```cpp
struct HandReplyData {
    sequence <HandReplyParam> hand_data;
    int32 hand_index;  // 0: left, 1: right
    int32 hand_type;   // 0: gripper, 1: dexterous hand
};
```

## Status Codes
The `status` field in `HandReplyParam` can have the following values:
- **0:** Opening
- **1:** Closing  
- **2:** Halt on target position
- **3:** Halt on target force
- **5:** Current Protection
- **6:** Motor Stall Protection
- **7:** Motor Error

## Error Codes
The `error` field in `HandReplyParam` uses bit flags:
- **bit0:** Stall Error
- **bit1:** Over Temperature  
- **bit2:** Over Current
- **bit3:** Motor Fault
- **bit4:** Communication Fault

## Usage Notes

### Current Monitoring
- The `current` field provides joint current feedback in milliamps (mA)
- Range: 0-1000 mA
- Use for monitoring motor load and detecting potential issues

### Temperature Monitoring  
- The `temp` field provides joint temperature in Celsius
- Range: 0-100 °C
- Monitor to prevent overheating and ensure safe operation

### Hand Identification
- Use `hand_index` to distinguish between left (0) and right (1) hands
- Use `hand_type` to distinguish between gripper (0) and dexterous hand (1)

### Error Handling
- Check the `error` field regularly for fault conditions
- Implement appropriate responses for each error type
- Monitor `status` field to understand current hand state