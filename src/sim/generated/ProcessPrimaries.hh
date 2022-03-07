#include "base/Assert.hh"
#include "base/Macros.hh"
#include "sim/detail/ProcessPrimariesLauncher.hh"
#include "base/Span.hh"
#include "physics/base/Primary.hh"
#include "sim/TrackInitData.hh"

namespace celeritas
{
namespace generated
{

void process_primaries(
    const Span<const Primary> primaries,
    const TrackInitStateHostRef& data);

void process_primaries(
    const Span<const Primary> primaries,
    const TrackInitStateDeviceRef& data);

#if !CELER_USE_DEVICE
inline void process_primaries(const Span<const Primary>, const TrackInitStateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

} // namespace generated
} // namespace celeritas
