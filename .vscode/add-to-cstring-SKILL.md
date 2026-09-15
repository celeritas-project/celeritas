---
name: add-to-cstring
description: Add a to_cstring function for a C++ enum type in the Celeritas codebase. Use when asked to add string conversion, to_cstring, or enum-to-string functionality for any enum class. Follows the standard EnumStringMapper pattern used throughout the codebase.
---

# Adding to_cstring for Enums

## When to Use
- User asks to add `to_cstring` for an enum
- Adding string conversion or enum-to-string functionality
- Need to make an enum printable/debuggable

## Pattern

### 1. Header File (.hh)
Add declaration after the enum definition, typically under a `// HELPER FUNCTIONS (HOST)` comment section:

```cpp
enum class MyEnum
{
    value1,
    value2,
    value3,
    size_,  // if present
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS (HOST)
//---------------------------------------------------------------------------//

// Get a string corresponding to [description of enum]
char const* to_cstring(MyEnum);
```

### 2. Implementation File (.cc)
Add implementation using `EnumStringMapper`:

```cpp
#include "corecel/io/EnumStringMapper.hh"  // Add this include

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to [description of enum].
 */
char const* to_cstring(MyEnum value)
{
    static EnumStringMapper<MyEnum> const to_cstring_impl{
        "value1",
        "value2",
        "value3",
    };
    return to_cstring_impl(value);
}
```

### Important Notes

1. **String order matters**: The strings must appear in the **same order** as the enum values
2. **Exclude size_**: Do NOT include a string for the `size_` sentinel value (if present)
3. **Use lowercase snake_case**: String values should match enum names exactly (typically lowercase with underscores)
4. **Add include**: Must include `"corecel/io/EnumStringMapper.hh"` in the .cc file
5. **Doxygen comment**: Implementation should have a doxygen comment describing what it returns

## Examples in Codebase

- `src/celeritas/Types.{hh,cc}`: `TrackStatus`, `MatterState`, `TrackOrder`
- `src/celeritas/g4/StateDependent.{hh,cc}`: `GeantStateChange`
- `src/orange/OrangeTypes.{hh,cc}`: `Sense`, `SurfaceType`, `TransformType`
- `src/corecel/Types.{hh,cc}`: `MemSpace`, `UnitSystem`

## Common Locations

- **Header section**: Look for existing `// HELPER FUNCTIONS (HOST)` sections or add one after type definitions
- **Implementation location**: Add near other `to_cstring` implementations at the end of the .cc file

## Testing

No unit tests are typically required for `to_cstring` functions unless the enum has special significance. The function is usually tested implicitly through usage in output/logging code.
