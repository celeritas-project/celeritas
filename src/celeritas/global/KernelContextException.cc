//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/KernelContextException.cc
//---------------------------------------------------------------------------//
#include "KernelContextException.hh"

#include "corecel/OpaqueIdIO.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/JsonPimpl.hh"
#if CELERITAS_USE_JSON
#    include "corecel/cont/ArrayIO.json.hh"
#    include "corecel/math/QuantityIO.json.hh"
#endif

#include "CoreTrackView.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
#if CELERITAS_USE_JSON
template<class V, class S>
void insert_if_valid(char const* key,
                     OpaqueId<V, S> const& val,
                     nlohmann::json* obj)
{
    if (val)
    {
        (*obj)[key] = val.unchecked_get();
    }
}
#endif

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with track data and kernel label.
 */
KernelContextException::KernelContextException(
    HostCRef<CoreParamsData> const& params,
    HostRef<CoreStateData> const& states,
    ThreadId thread,
    std::string&& label)
    : thread_(thread), label_(std::move(label))
{
    try
    {
        if (!thread)
        {
            // Make sure the thread is valid before trying to construct
            // detailed debug information
            throw std::exception();
        }
        CoreTrackView core(params, states, thread);
        this->initialize(core);
    }
    catch (...)
    {
        // Ignore all exceptions while trying to process diagnostic information
        what_ = label_ + " (error processing track state)";
    }
}

//---------------------------------------------------------------------------//
/*!
 * This class type's description.
 */
char const* KernelContextException::type() const
{
    return "KernelContextException";
}

//---------------------------------------------------------------------------//
/*!
 * Save context to a JSON object.
 */
void KernelContextException::output(JsonPimpl* json) const
{
#if CELERITAS_USE_JSON
    nlohmann::json j;
#    define KCE_INSERT_IF_VALID(ATTR) insert_if_valid(#ATTR, ATTR##_, &j)

    KCE_INSERT_IF_VALID(thread);
    KCE_INSERT_IF_VALID(track_slot);
    KCE_INSERT_IF_VALID(event);
    KCE_INSERT_IF_VALID(track);
    if (track_)
    {
        KCE_INSERT_IF_VALID(parent);
        j["num_steps"] = num_steps_;
        KCE_INSERT_IF_VALID(particle);
        j["energy"] = energy_;
        j["pos"] = pos_;
        j["dir"] = dir_;
        KCE_INSERT_IF_VALID(volume);
        KCE_INSERT_IF_VALID(surface);
    }
    if (!label_.empty())
    {
        j["label"] = label_;
    }
#    undef KCE_INSERT_IF_VALID
    json->obj = std::move(j);
#else
    CELER_DISCARD(json);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Populate properties during construction.
 */
void KernelContextException::initialize(CoreTrackView const& core)
{
    track_slot_ = core.track_slot_id();
    auto const&& sim = core.make_sim_view();
    if (sim.status() == TrackStatus::alive)
    {
        event_ = sim.event_id();
        track_ = sim.track_id();
        parent_ = sim.parent_id();
        num_steps_ = sim.num_steps();
        {
            auto const&& par = core.make_particle_view();
            particle_ = par.particle_id();
            energy_ = par.energy();
        }
        {
            auto const&& geo = core.make_geo_view();
            pos_ = geo.pos();
            dir_ = geo.dir();
            volume_ = geo.volume_id();
            surface_ = geo.surface_id();
        }
    }
    {
        // Construct std::exception message
        std::ostringstream os;
        os << "kernel context: track slot " << track_slot_ << " in '" << label_
           << "'";
        if (track_)
        {
            os << ", track " << track_ << " of event " << event_;
        }
        what_ = os.str();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
