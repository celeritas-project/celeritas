#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"


namespace celeritas
{
namespace generated
{

void process_secondaries(
    CoreHostRef const& core_data);

void process_secondaries(
    CoreDeviceRef const& core_data);

#if !CELER_USE_DEVICE
inline void process_secondaries(CoreDeviceRef const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

}  // namespace generated
}  // namespace celeritas
