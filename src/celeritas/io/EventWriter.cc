//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventWriter.cc
//---------------------------------------------------------------------------//
#include "EventWriter.hh"

#include <set>
#include <HepMC3/GenParticle.h>
#include <HepMC3/WriterAscii.h>
#include <HepMC3/WriterAsciiHepMC2.h>
#include <HepMC3/WriterHEPEVT.h>

#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeAndRedirect.hh"
#include "corecel/io/StringEnumMapper.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
EventWriter::Format filename_to_format(std::string const& filename)
{
    std::string ext;
    auto pos = filename.rfind('.');
    if (pos != std::string::npos)
    {
        ext = filename.substr(pos + 1);
    }

    static auto const from_string
        = StringEnumMapper<EventWriter::Format>::from_cstring_func(
            to_cstring, "event file format");
    return from_string(ext);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct by parsing the extension.
 */
EventWriter::EventWriter(std::string const& filename, SPConstParticles params)
    : EventWriter{filename, params, filename_to_format(filename)}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a filename, particle data, and output format.
 */
EventWriter::EventWriter(std::string const& filename,
                         SPConstParticles particles,
                         Format fmt)
    : particles_(particles)
{
    CELER_EXPECT(!filename.empty());
    CELER_EXPECT(particles_);

    CELER_LOG(info) << "Creating " << to_cstring(fmt) << " event file at "
                    << filename;
    ScopedTimeAndRedirect temp_{"HepMC3"};

    if (fmt != Format::hepmc3)
    {
        CELER_LOG(warning) << "Writing to a event file format other than "
                              "hepmc3 may result in an unreadable file";
    }

    writer_ = [&]() -> std::shared_ptr<HepMC3::Writer> {
        switch (fmt)
        {
            case Format::hepevt:
                return std::make_shared<HepMC3::WriterHEPEVT>(filename);
            case Format::hepmc2:
                return std::make_shared<HepMC3::WriterAsciiHepMC2>(filename);
            case Format::hepmc3:
                return std::make_shared<HepMC3::WriterAscii>(filename);
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }();

    CELER_ENSURE(writer_);
}

//---------------------------------------------------------------------------//
/*!
 * Write all primaries from a single event.
 */
void EventWriter::operator()(argument_type primaries)
{
    CELER_EXPECT(!primaries.empty());

    std::set<EventId::size_type> mismatched_events;

    HepMC3::GenEvent evt;
    evt.set_units(HepMC3::Units::MEV, HepMC3::Units::CM);

    // Vertex and corresponding celeritas position
    HepMC3::GenVertexPtr vtx;
    Primary const* vtx_primary{nullptr};

    // Loop over all primaries
    for (Primary const& p : primaries)
    {
        auto par = std::make_shared<HepMC3::GenParticle>();

        par->set_pid(particles_->id_to_pdg(p.particle_id).get());
        par->set_status(4);  // "incoming beam particle" from HepMC2 manual

        if (!vtx_primary || vtx_primary->time != p.time
            || vtx_primary->position != p.position)
        {
            if (vtx)
            {
                evt.add_vertex(vtx);
            }
            // Position has changed: add a new vertex
            vtx_primary = &p;

            HepMC3::FourVector pos;
            pos.set_x(p.position[0]);
            pos.set_y(p.position[1]);
            pos.set_z(p.position[2]);
            pos.set_t(p.time / units::centimeter * constants::c_light);
            vtx = std::make_shared<HepMC3::GenVertex>(pos);
        }

        HepMC3::FourVector mom;
        mom.set_px(p.direction[0]);
        mom.set_py(p.direction[1]);
        mom.set_pz(p.direction[2]);
        mom.set_e(value_as<units::MevEnergy>(p.energy));
        par->set_momentum(mom);

        if (CELER_UNLIKELY(p.event_id != EventId{event_count_}))
        {
            mismatched_events.insert(p.event_id.unchecked_get());
        }

        // Note: primary's track ID is ignored

        // Note: the particle has to be added to both in *and* out to be
        // readable in hepmc3 format
        CELER_ASSERT(vtx);
        vtx->add_particle_in(par);
        vtx->add_particle_out(par);
    }
    evt.add_vertex(vtx);

    if (CELER_UNLIKELY(!mismatched_events.empty()))
    {
        CELER_LOG(warning)
            << "Overwriting primary event IDs with " << event_count_ << ": "
            << join(mismatched_events.begin(), mismatched_events.end(), ", ");
    }

    {
        ScopedTimeAndRedirect temp_{"HepMC3"};
        writer_->write_event(evt);
    }
    ++event_count_;
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a state of matter.
 */
char const* to_cstring(EventWriter::Format value)
{
    static EnumStringMapper<EventWriter::Format> const to_cstring_impl{
        "hepevt",
        "hepmc2",
        "hepmc3",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
