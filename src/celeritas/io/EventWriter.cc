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
#include <HepMC3/GenVertex.h>
#include <HepMC3/Print.h>
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
    : EventWriter{filename, std::move(params), filename_to_format(filename)}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a filename, particle data, and output format.
 */
EventWriter::EventWriter(std::string const& filename,
                         SPConstParticles particles,
                         Format fmt)
    : particles_{particles}, fmt_{fmt}
{
    CELER_EXPECT(!filename.empty());
    CELER_EXPECT(particles_);
    CELER_EXPECT(fmt_ != Format::size_);

    CELER_LOG(info) << "Creating " << to_cstring(fmt) << " event file at "
                    << filename;
    ScopedTimeAndRedirect temp_{"HepMC3"};

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

    HepMC3::GenEvent evt(HepMC3::Units::MEV, HepMC3::Units::CM);

    EventId const event_id{event_count_++};

    // Vertex and corresponding celeritas position
    HepMC3::GenVertexPtr vtx;
    Primary const* vtx_primary{nullptr};

    // See HepMC2 user manual (page 13): we only use 0 and 1
    enum StatusCode
    {
        meaningless_code = 0,
        final_code = 1,
        decayed_code = 2,
        documentation_code = 3,
        beam_code = 4,
    };

    // Loop over all primaries
    for (Primary const& p : primaries)
    {
        if (!vtx_primary || vtx_primary->time != p.time
            || vtx_primary->position != p.position)
        {
            // Position or time has changed: add a new vertex
            vtx_primary = &p;

            HepMC3::FourVector pos;
            pos.set_x(p.position[0]);
            pos.set_y(p.position[1]);
            pos.set_z(p.position[2]);
            pos.set_t(p.time / units::centimeter * constants::c_light);

            // Use the last particle from the previous vertex as an "in"
            // particle (needed for the file format to be correct)
            auto last_par = [&vtx] {
                if (!vtx)
                {
                    // No 'primary' particle exists: fake one
                    auto result = std::make_shared<HepMC3::GenParticle>();
                    result->set_status(meaningless_code);
                    return result;
                }
                CELER_ASSERT(!vtx->particles_out().empty());
                return vtx->particles_out().back();
            }();
            vtx = std::make_shared<HepMC3::GenVertex>(pos);
            vtx->add_particle_in(last_par);
            evt.add_vertex(vtx);
        }

        CELER_ASSERT(vtx);
        auto par = std::make_shared<HepMC3::GenParticle>();
        vtx->add_particle_out(par);

        par->set_pid(particles_->id_to_pdg(p.particle_id).get());
        par->set_status(final_code);

        HepMC3::FourVector mom;
        mom.set_px(p.direction[0]);
        mom.set_py(p.direction[1]);
        mom.set_pz(p.direction[2]);
        mom.set_e(value_as<units::MevEnergy>(p.energy));
        par->set_momentum(mom);

        if (CELER_UNLIKELY(p.event_id != event_id))
        {
            mismatched_events.insert(p.event_id.unchecked_get());
        }

        // Note: primary's track ID is ignored
    }

    if (CELER_UNLIKELY(!mismatched_events.empty()))
    {
        CELER_LOG_LOCAL(warning)
            << "Overwriting primary event IDs with " << event_id.get() << ": "
            << join(mismatched_events.begin(), mismatched_events.end(), ", ");
    }
    evt.set_event_number(event_id.get());

    if (fmt_ == Format::hepevt)
    {
        // HEPEVT files can only write in GeV/mm
        evt.set_units(HepMC3::Units::GEV, HepMC3::Units::MM);
    }

    {
        ScopedTimeAndRedirect temp_{"HepMC3"};
        writer_->write_event(evt);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to an event record format.
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
