#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "corecel/cont/Span.hh"
#include "celeritas/phys/Primary.hh"

namespace celeritas
{
namespace generated
{

void process_primaries(
    const CoreHostRef& core_data,
    const Span<const Primary> primaries);

void process_primaries(
    const CoreDeviceRef& core_data,
    const Span<const Primary> primaries);

#if !CELER_USE_DEVICE
inline void process_primaries(const CoreDeviceRef&, const Span<const Primary>)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

} // namespace generated
} // namespace celeritas
