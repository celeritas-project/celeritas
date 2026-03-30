//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalStepGatherExecutor.hh
//---------------------------------------------------------------------------//
#pragma once
#include <fstream>
#include <nlohmann/json.hpp>

#include "geocel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/CoreTrackView.hh"

#include "OpticalStepData.hh"
#include "OpticalStepParams.hh"

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
    OpticalStepGatherExecutor ...;
   \endcode
 */
template<StepPoint P>
struct OpticalStepGatherExecutor
{
    NativeCRef<CoreParamsData> const& params;
    NativeRef<CoreStateData> const& states;
    NativeRef<OpticalStepStateData> step;

    CELER_FUNCTION void operator()(CoreTrackView& track) const
    {
        // CoreTrackView track(params, states, tid);

        if (!is_track_valid(track.sim().status()))
            return;
        auto tid = track.track_slot_id();
        auto idx = tid.get();
        auto geo = track.geometry();
        auto sim = track.sim();
        auto mat = track.material_record();
        auto part = track.particle();
        auto const& direction = geo.dir();

        static std::ofstream out("optical_steps.jsonl", std::ios::app);
        static std::mutex file_mutex;
        static unsigned int global_event_id = 0;

        auto const& pos = geo.pos();

        if constexpr (P == StepPoint::pre)
        {
            if (sim.num_steps() == 1)
            {
                nlohmann::json j;
                unsigned int event_id = 0;
                {
                    std::lock_guard<std::mutex> lock(file_mutex);
                    event_id = ++global_event_id;
                }
                j["event_id"] = event_id;
                j["track"] = idx;
                j["volume"] = geo.volume_id().unchecked_get();
                j["step"] = -1;
                // j["x"] = pos[0];
                // j["y"] = pos[1];
                // j["z"] = pos[2];
                j["energy"] = part.energy().value();
                // j["post_action"] = sim.post_step_action().unchecked_get();
                // j["dir_x"] = direction[0];
                // j["dir_y"] = direction[1];
                // j["dir_z"] = direction[2];
                {
                    std::lock_guard<std::mutex> lock(file_mutex);
                    out << j.dump() << "\n";
                }
            }
        }

        else if constexpr (P == StepPoint::post)
        {
            nlohmann::json j;
            unsigned int event_id = 0;
            {
                std::lock_guard<std::mutex> lock(file_mutex);
                event_id = ++global_event_id;
            }
            j["event_id"] = event_id;
            j["track"] = idx;
            j["volume"] = geo.volume_id().unchecked_get();
            j["step"] = sim.num_steps();
            // j["x"] = pos[0];
            // j["y"] = pos[1];
            // j["z"] = pos[2];
            j["energy"] = part.energy().value();
            // j["post_action"] = sim.post_step_action().unchecked_get();
            // j["dir_x"] = direction[0];
            // j["dir_y"] = direction[1];
            // j["dir_z"] = direction[2];
            {
                std::lock_guard<std::mutex> lock(file_mutex);
                out << j.dump() << "\n";
            }

            CELER_LOG(debug) << "Event ID: " << event_id << " | Volume ID : "
                             << track.geometry().volume_id().unchecked_get()
                             << " Track id : " << tid << " at x = " << pos[0]
                             << "y = " << pos[1] << " z = " << pos[2]
                             << " energy = " << part.energy().value()
                             << " Post step action id :  "
                             << sim.post_step_action().unchecked_get()
                             << " Direction : " << direction[0] << " y "
                             << direction[1] << " z " << direction[2];
        }
    }
};

}  // namespace detail
}  // namespace optical
}  // namespace celeritas
