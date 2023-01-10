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
    CoreHostRef const& core_data,
    Span<const Primary> const primaries);

void process_primaries(
    CoreDeviceRef const& core_data,
    Span<const Primary> const primaries);

#if !CELER_USE_DEVICE
inline void process_primaries(CoreDeviceRef const&, Span<const Primary> const)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

}  // namespace generated
}  // namespace celeritas
