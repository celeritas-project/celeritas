//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalStepData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    OpticalStepData ...;
   \endcode
 */

//---------------------------------------------
// PARAM DATA
//---------------------------------------------
template<Ownership W, MemSpace M>
struct OpticalStepParamsData
{
    explicit CELER_FUNCTION operator bool() const { return true; }

    template<Ownership W2, MemSpace M2>
    OpticalStepParamsData& operator=(OpticalStepParamsData<W2, M2> const&)
    {
        return *this;
    }
};

//---------------------------------------------
// STATE DATA
//---------------------------------------------
template<Ownership W, MemSpace M>
struct OpticalStepStateData
{
    template<class T>
    using StateItems = StateCollection<T, W, M>;

    StreamId stream;

    StateItems<TrackSlotId> track_slot;
    StateItems<Real3> pos;
    StateItems<VolumeId> volume_id;

    CELER_FUNCTION size_type size() const { return track_slot.size(); }

    explicit CELER_FUNCTION operator bool() const
    {
        return stream && !track_slot.empty();
    }

    template<Ownership W2, MemSpace M2>
    OpticalStepStateData& operator=(OpticalStepStateData<W2, M2>& other)
    {
        stream = other.stream;
        track_slot = other.track_slot;
        pos = other.pos;
        volume_id = other.volume_id;
        return *this;
    }
};

template<MemSpace M>
inline void resize(OpticalStepStateData<Ownership::value, M>* state,
                   HostCRef<OpticalStepParamsData> const&,
                   StreamId sid,
                   size_type count)
{
    //  CELER_LOG(info) << "Optical step state" << count;
    CELER_LOG(info) << "Resizing OpticalStepState to " << count;
    state->stream = sid;

    resize(&state->track_slot, count);
    resize(&state->pos, count);
    resize(&state->volume_id, count);
}

struct OpticalStepRecord
{
    unsigned int slot{};
    unsigned int volume{};
    unsigned int material{};
    unsigned int action{};
    double energy{};
    double pos[3]{};
};

inline void to_json(nlohmann::json& j, OpticalStepRecord const& r)
{
    j = {{"slot", r.slot},
         {"volume", r.volume},
         {"material", r.material},
         {"action", r.action},
         {"energy", r.energy},
         {"pos", {r.pos[0], r.pos[1], r.pos[2]}}};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
