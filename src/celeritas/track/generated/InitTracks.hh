#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"


namespace celeritas
{
namespace generated
{

void init_tracks(
    CoreHostRef const& core_data,
    size_type const num_vacancies);

void init_tracks(
    CoreDeviceRef const& core_data,
    size_type const num_vacancies);

#if !CELER_USE_DEVICE
inline void init_tracks(CoreDeviceRef const&, size_type const)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

}  // namespace generated
}  // namespace celeritas
