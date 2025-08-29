#pragma once

#include "celeritas/optical/CoreTrackView.hh"

namespace celeritas
{
namespace optical
{

struct FakeExecutor
{
    CELER_FUNCTION void operator()(CoreTrackView& track) const
    {
        CELER_ASSERT(track.surface_physics().is_crossing_boundary());
    }
};

}  // namespace optical
}  // namespace celeritas
