#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/track/detail/ProcessPrimariesLauncher.hh"
#include "corecel/cont/Span.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/track/TrackInitData.hh"

namespace celeritas
{
namespace generated
{

void process_primaries(
    const Span<const Primary> primaries,
    const TrackInitStateHostRef& init_data);

void process_primaries(
    const Span<const Primary> primaries,
    const TrackInitStateDeviceRef& init_data);

#if !CELER_USE_DEVICE
inline void process_primaries(const Span<const Primary>, const TrackInitStateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

} // namespace generated
} // namespace celeritas
