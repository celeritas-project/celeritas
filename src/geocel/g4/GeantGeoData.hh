//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeoData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/Types.hh"

#include "detail/GeantGeoNavCollection.hh"

class G4VPhysicalVolume;

namespace celeritas
{
namespace detail
{
class GeantVolumeInstanceMapper;
}
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Geant4 data is all global.
 */
struct GeantGeoParamsDataBase
{
    //! Pointer to the Geant4 world
    G4VPhysicalVolume* world{nullptr};

    //! Instance ID of the first logical volume in the store
    ImplVolumeId::size_type lv_offset{0};
    //! Instance ID of the first material in the store
    GeoMatId::size_type mat_offset{0};
    //! Instance mapper owned by GeantGeoParams
    detail::GeantVolumeInstanceMapper const* vi_mapper{nullptr};

    //! Navigator verbosity
    int nav_verbosity_{0};

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return world != nullptr && vi_mapper != nullptr;
    }
};

template<Ownership W, MemSpace M>
struct GeantGeoParamsData : GeantGeoParamsDataBase
{
    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    GeantGeoParamsData& operator=(GeantGeoParamsData<W2, M2>&)
    {
        // We always build "host refs" in GeantGeoParams
        CELER_ASSERT_UNREACHABLE();
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Geant4 geometry state data.
 */
template<Ownership W, MemSpace M>
struct GeantGeoStateData
{
    //// TYPES ////

    using real_type = double;
    using Real3 = Array<double, 3>;
    template<class T>
    using StateItems = celeritas::StateCollection<T, W, MemSpace::host>;

    //// DATA ////

    // Collections
    StateItems<Real3> pos;
    StateItems<Real3> dir;
    StateItems<Real3> normal;
    StateItems<real_type> next_step;
    StateItems<real_type> safety_radius;
    StateItems<GeoStatus> status;

    // Wrapper for G4TouchableHistory and G4Navigator
    detail::GeantGeoNavCollection<W, M> nav_state;

    //// METHODS ////

    //! True if sizes are consistent and states are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return this->size() > 0 && dir.size() == this->size()
               && normal.size() == this->size()
               && next_step.size() == this->size()
               && safety_radius.size() == this->size()
               && status.size() == this->size()
               && nav_state.size() == this->size();
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const { return pos.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    GeantGeoStateData& operator=(GeantGeoStateData<W2, M2>& other)
    {
        static_assert(M2 == M && W == Ownership::reference,
                      "Only supported assignment is from the same memspace to "
                      "a reference");
        CELER_EXPECT(other);
        pos = other.pos;
        dir = other.dir;
        normal = other.normal;
        next_step = other.next_step;
        safety_radius = other.safety_radius;
        status = other.status;
        nav_state = other.nav_state;
        return *this;
    }

    void reset() { nav_state.reset(); }
};

//---------------------------------------------------------------------------//
/*!
 * Resize geometry states.
 */
inline void resize(GeantGeoStateData<Ownership::value, MemSpace::host>* data,
                   HostCRef<GeantGeoParamsData> const& params,
                   StreamId stream_id,
                   size_type size)
{
    CELER_EXPECT(data);
    CELER_EXPECT(size > 0);

    resize(&data->pos, size);
    resize(&data->dir, size);
    resize(&data->normal, size);
    resize(&data->next_step, size);
    resize(&data->safety_radius, size);
    resize(&data->status, size);
    data->nav_state.resize(params, stream_id, size);

    CELER_ENSURE(data);
}

//! Resize assuming stream zero for serial tests.
inline void resize(GeantGeoStateData<Ownership::value, MemSpace::host>* data,
                   HostCRef<GeantGeoParamsData> const& params,
                   size_type size)
{
    return resize(data, params, StreamId{0}, size);
}

inline void resize(GeantGeoStateData<Ownership::value, MemSpace::device>*,
                   HostCRef<GeantGeoParamsData> const&,
                   StreamId,
                   size_type)
{
    CELER_NOT_IMPLEMENTED("Geant4 GPU");
}

inline void resize(GeantGeoStateData<Ownership::value, MemSpace::device>*,
                   HostCRef<GeantGeoParamsData> const&,
                   size_type)
{
    CELER_NOT_IMPLEMENTED("Geant4 GPU");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
