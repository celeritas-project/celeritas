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
    HostCRef<CoreParamsData> const& core_params,
    HostRef<CoreStateData> const& core_states,
    Span<const Primary> const primaries);

void process_primaries(
    DeviceCRef<CoreParamsData> const& core_params,
    DeviceRef<CoreStateData> const& core_states,
    Span<const Primary> const primaries);

#if !CELER_USE_DEVICE
inline void process_primaries(DeviceCRef<CoreParamsData> const&, DeviceRef<CoreStateData> const&, Span<const Primary> const)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

}  // namespace generated
}  // namespace celeritas
