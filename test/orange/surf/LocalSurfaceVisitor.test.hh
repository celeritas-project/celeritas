//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/LocalSurfaceVisitor.test.hh
//---------------------------------------------------------------------------//
#include <vector>

#include "corecel/Macros.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "orange/OrangeData.hh"
#include "orange/surf/LocalSurfaceVisitor.hh"

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
/*!
 * Small subset of state for testing.
 */
template<Ownership W, MemSpace M>
struct OrangeMiniStateData
{
    template<class T>
    using StateItems = StateCollection<T, W, M>;

    StateItems<Real3> pos;
    StateItems<Real3> dir;
    StateItems<Sense> sense;
    StateItems<real_type> distance;

    //! True if sizes are consistent and nonzero
    explicit CELER_FUNCTION operator bool() const
    {
        // clang-format off
        return pos.size() > 0
            && dir.size() == pos.size()
            && sense.size() == pos.size()
            && distance.size() == pos.size();
        // clang-format on
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OrangeMiniStateData& operator=(OrangeMiniStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        pos = other.pos;
        dir = other.dir;
        sense = other.sense;
        distance = other.distance;
        CELER_ENSURE(*this);
        return *this;
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const { return pos.size(); }
};

//---------------------------------------------------------------------------//
/*!
 * Resize states in host code.
 */
template<MemSpace M>
inline void resize(OrangeMiniStateData<Ownership::value, M>* data,
                   HostCRef<OrangeParamsData> const&,
                   size_type size)
{
    CELER_EXPECT(data);
    CELER_EXPECT(size > 0);
    resize(&data->pos, size);
    resize(&data->dir, size);
    resize(&data->sense, size);
    resize(&data->distance, size);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
//! Calculate the sense of a surface at a given position.
struct CalcSenseDistance
{
    Real3 const& pos;
    Real3 const& dir;
    Sense* sense;
    real_type* distance;

    template<class S>
    CELER_FUNCTION void operator()(S&& surf)
    {
        // Calculate sense
        auto signed_sense = surf.calc_sense(this->pos);
        *this->sense = to_sense(signed_sense);

        // Calculate nearest distance
        auto intersect = surf.calc_intersections(
            this->pos, this->dir, to_surface_state(signed_sense));
        for (real_type distance : intersect)
        {
            CELER_ASSERT(distance > 0);
        }
        *this->distance = *min_element(intersect.begin(), intersect.end());
    }
};

//! Calculate distance to a single surface
template<MemSpace M = MemSpace::native>
struct CalcSenseDistanceExecutor
{
    OrangeParamsData<Ownership::const_reference, M> params;
    OrangeMiniStateData<Ownership::reference, M> states;

    CELER_FUNCTION void operator()(TrackSlotId tid) const
    {
        CELER_EXPECT(this->params.simple_units.size() == 1);
        LocalSurfaceVisitor visit(this->params, SimpleUnitId{0});
        CalcSenseDistance calc_sense_dist{this->states.pos[tid],
                                          this->states.dir[tid],
                                          &this->states.sense[tid],
                                          &this->states.distance[tid]};
        auto num_surf = params.simple_units[SimpleUnitId{0}].surfaces.size();
        visit(calc_sense_dist, LocalSurfaceId{tid.get() % num_surf});
    }
};

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct SATestInput
{
    using ParamsRef = DeviceCRef<OrangeParamsData>;
    using StateRef = DeviceRef<OrangeMiniStateData>;

    ParamsRef params;
    StateRef states;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
void sa_test(SATestInput);

#if !CELER_USE_DEVICE
inline void sa_test(SATestInput)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
