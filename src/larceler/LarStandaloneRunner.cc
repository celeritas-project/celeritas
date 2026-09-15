//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarStandaloneRunner.cc
//---------------------------------------------------------------------------//
#include "LarStandaloneRunner.hh"

#include <limits>
#include <memory>
#include <utility>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>
#include <lardataobj/Simulation/SimPhotons.h>
#include <lardataobj/Simulation/sim.h>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stopwatch.hh"
#include "geocel/DetectorParams.hh"  // IWYU pragma: keep
#include "geocel/VolumeParams.hh"  // IWYU pragma: keep
#include "geocel/detail/LengthUnits.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/CoreGeoParams.hh"  // IWYU pragma: keep
#include "celeritas/inp/StandaloneInput.hh"  // IWYU pragma: keep
#include "celeritas/io/OpticalDistributionWriter.hh"  // IWYU pragma: keep
#include "celeritas/optical/CoreParams.hh"  // IWYU pragma: keep
#include "celeritas/optical/Runner.hh"

#include "Convert.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Hide the goofy lvalue "move" implementation of OBTR
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
CELER_FORCEINLINE auto make_obtr(sim::OBTRHelper&& helper)
{
    return sim::OpDetBacktrackerRecord(helper);
}

//! Starting index of a track ID from LArSoft that has a negative value
constexpr auto neg_trackid_offset{
    std::numeric_limits<PrimaryId::size_type>::max() / 2u};

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with problem setup and detector ID coordinates.
 *
 * The detector "channels" (coordinates) should be input as a vector.
 */
LarStandaloneRunner::LarStandaloneRunner(Input&& i, VecReal3 const& det_coords)
{
    CELER_EXPECT(!det_coords.empty());
    CELER_EXPECT(!i.detectors.empty());

    CELER_LOG(info) << "Setting up Celeritas optical standalone runner built "
                       "against LArSoft v"
                    << cmake::larsoft_version << " components";

    i.problem.detectors.callback
        = [this](SpanCelerHits h) { return this->hit(h); };
    runner_ = std::make_shared<optical::Runner>(std::move(i));

    ScopedProfiling profile_this("setup-channels");
    // Map detector coordinates
    auto geo = runner_->params()->geometry();
    CELER_ASSERT(geo);
    auto vols = runner_->params()->volume();
    CELER_ASSERT(vols);
    auto dets = runner_->params()->detectors();
    CELER_ASSERT(dets);

    geo_to_channel_.reserve(det_coords.size());
    btr_helpers_.reserve(det_coords.size());
    lite_hits_.reserve(det_coords.size());
    for (auto i : range(det_coords.size()))
    {
        auto inst_id = geo->find_volume_instance_at(det_coords[i]);
        CELER_VALIDATE(inst_id,
                       << "could not find a volume at " << det_coords[i]
                       << " [" << lengthunits::native_label << "]");
        auto&& [iter, inserted] = geo_to_channel_.insert({inst_id, i});
        CELER_VALIDATE(inserted,
                       << "multiple detector IDs (" << i << ", "
                       << iter->second << ") share the same volume instance ("
                       << vols->volume_instance_labels().at(inst_id));
    }
    CELER_ENSURE(this->num_channels() == det_coords.size());
}

//---------------------------------------------------------------------------//
/*!
 * Run scintillation optical photons from a single set of energy steps.
 *
 * The optical material is determined in Celeritas when the tracks are
 * initialized from the pre-step position.
 *
 * Note that the returned BTRs *omit* empty channels, just as PDFastSimPar does.
 *
 * \todo With Cherenkov enabled we need the incident particle's charge and the
 * pre- and post-step speed.
 */
auto LarStandaloneRunner::operator()(VecSED const& sim_energy_deposits)
    -> result_type
{
    CELER_EXPECT(!sim_energy_deposits.empty());
    CELER_EXPECT(lite_hits_.empty());

    // Allocate BTR helpers
    btr_helpers_.clear();
    for (auto i : range(this->num_channels()))
    {
        btr_helpers_.emplace_back(std::make_unique<sim::OBTRHelper>(i));
    }
    // Clear any existing lite hits
    lite_hits_.resize(this->num_channels());
    for (auto& lh : lite_hits_)
    {
        lh.clear();
    }

    // Convert SimEnergyDep input and save metadata for BTRs
    std::vector<celeritas::optical::GeneratorDistributionData> gdd;
    gdd.reserve(sim_energy_deposits.size());
    step_md_.clear();
    step_md_.reserve(sim_energy_deposits.size());
    size_type num_skipped{0};
    double edep_skipped{0};
    for (auto i : range(sim_energy_deposits.size()))
    {
        auto const& edepi = sim_energy_deposits[i];
        if (edepi.NumPhotons() == 0)
        {
            ++num_skipped;
            edep_skipped += edepi.Energy();
            continue;
        }

        // Convert LArSoft sim edeps to Celeritas generator distribution data
        // TODO: use individual fast/slow spectra by multiplying with the
        // edep's ScintYieldRatio() (fraction fast spectrum)
        celeritas::optical::GeneratorDistributionData data;
        data.type = GeneratorType::scintillation;
        data.num_photons = edepi.NumPhotons();
        data.step_length = convert_from_larsoft<LarsoftLen>(edepi.StepLength());
        // Assume continuous energy loss along the step
        //! \todo For neutral particles, set this to 0 (LED at post-step point)
        data.continuous_edep_fraction = 1;
        data.points[StepPoint::pre].time
            = convert_from_larsoft<LarsoftTime>(edepi.StartT());
        data.points[StepPoint::pre].pos
            = convert_from_larsoft<LarsoftLen>(edepi.Start());
        data.points[StepPoint::post].time
            = convert_from_larsoft<LarsoftTime>(edepi.EndT());
        data.points[StepPoint::post].pos
            = convert_from_larsoft<LarsoftLen>(edepi.End());
        data.primary = id_cast<PrimaryId>(step_md_.size());
        CELER_ASSERT(data);
        gdd.push_back(data);

        step_md_.push_back([&edepi] {
            StepMetadata md;
            md.track_id = edepi.TrackID();
            md.midpoint[0] = edepi.MidPointX();
            md.midpoint[1] = edepi.MidPointY();
            md.midpoint[2] = edepi.MidPointZ();
            md.avg_edep = edepi.Energy() / edepi.NumPhotons();
            CELER_ENSURE(md.avg_edep > 0);
            CELER_ENSURE(md.track_id != sim::NoParticleId);
            return md;
        }());
    }
    if (num_skipped > 0)
    {
        CELER_LOG(warning)
            << "Omitting " << num_skipped
            << " steps that emitted zero photons (total energy deposition: "
            << edep_skipped << " MeV)";
    }

    if (gdd.empty())
    {
        CELER_LOG(warning) << "No energy deposition resulted in photons: "
                              "skipping optical transport";
        return {};
    }

    if (runner_->problem().offload_writer)
    {
        // Dump distribution data to a file
        (*runner_->problem().offload_writer)(gdd);
    }

    // Execute
    runner_->insert(make_span(std::as_const(gdd)));
    Stopwatch get_transport_time;
    auto result = (*runner_)();

    CELER_ASSERT(result.counters.generators.size() == 1);
    auto const& gen = result.counters.generators.front();
    CELER_LOG(debug) << "Transported " << gen.num_generated
                     << " optical photons from " << gen.buffer_size
                     << " sim energy deposits with a total of "
                     << result.counters.steps << " steps over "
                     << result.counters.step_iters << " step iterations in "
                     << get_transport_time() << "s";

    // Convert BTR helpers to BTRs in the LarSoft order
    // and convert SimPhotons from unordered to ordered map
    VecSPL sim_photons(lite_hits_.size());
    VecBTR btrs;
    btrs.reserve(btr_helpers_.size());
    for (auto channel_id : range(lite_hits_.size()))
    {
        // Convert photon hits
        auto& lite = lite_hits_[channel_id];
        auto& spl = sim_photons[channel_id];
        spl.OpChannel = channel_id;
        for (auto&& kv : lite)
        {
            spl.DetectedPhotons.insert(kv);
        }

        if (!lite.empty())
        {
            // Only insert BTRs if optical hits were encountered (for
            // consistency with PDFastSimPar)
            btrs.emplace_back(make_obtr(std::move(*btr_helpers_[channel_id])));
            CELER_ENSURE(btrs.back().OpDetNum()
                         == static_cast<int>(channel_id));
        }
    }

    CELER_ENSURE(sim_photons.size() == this->num_channels());
    CELER_ENSURE(btrs.size() <= this->num_channels());
    return result_type{std::move(sim_photons), std::move(btrs)};
}

//---------------------------------------------------------------------------//
/*!
 * Convert Celeritas hits to optical backtracker records.
 */
void LarStandaloneRunner::hit(SpanCelerHits hits)
{
    CELER_LOG_LOCAL(debug) << "Processing " << hits.size() << " hits";

    for (auto& h : hits)
    {
        CELER_ASSERT(h.volume_instance);
        auto iter = geo_to_channel_.find(h.volume_instance);
        CELER_ASSERT(iter != geo_to_channel_.end());
        unsigned int det_id = iter->second;
        CELER_ASSERT(det_id < lite_hits_.size()
                     && det_id < btr_helpers_.size());
        CELER_ASSERT(h.primary < step_md_.size());
        auto const& step_md = step_md_[*h.primary];

        // BTR position is *emission point* not *hit point*:
        // TODO: to get additional accuracy, we can record the exact
        // emission point of each photon and carry it along with the track
        // This may need to be justified by reconstruction
        btr_helpers_[det_id]->AddScintillationPhotonsToMap(
            step_md.track_id,
            convert_to_larsoft<LarsoftTime>(h.time),
            /* num photons = */ 1,
            step_md.midpoint.data(),
            step_md.avg_edep);

        // Add to sim photon hits, rounding to nearest nanosecond
        // TODO: if WLS is used for cryostat reflection, we'll need
        // separate arrays [see e.g.,
        // produces<std::vector<sim::SimPhotonsLite>>("Reflected") in
        // PDFastSimPAR] and probably discriminate based on the wavelength
        // of the detected photon
        auto rounded_time_ns
            = std::round(convert_to_larsoft<LarsoftTime>(h.time));
        CELER_ASSERT(rounded_time_ns > 0
                     && rounded_time_ns < static_cast<double>(
                            std::numeric_limits<int>::max()));
        ++lite_hits_[det_id][rounded_time_ns];
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
