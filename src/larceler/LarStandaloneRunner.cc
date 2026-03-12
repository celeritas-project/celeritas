//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarStandaloneRunner.cc
//---------------------------------------------------------------------------//
#include "LarStandaloneRunner.hh"

#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayQuantity.hh"
#include "geocel/DetectorParams.hh"  // IWYU pragma: keep
#include "geocel/Types.hh"
#include "geocel/VolumeParams.hh"  // IWYU pragma: keep
#include "geocel/detail/LengthUnits.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/CoreGeoParams.hh"  // IWYU pragma: keep
#include "celeritas/inp/StandaloneInput.hh"  // IWYU pragma: keep
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

//---------------------------------------------------------------------------//
/*!
 * Convert LArSoft track ID to Celeritas primary ID.
 *
 * Split unsigned range: [0, half) for non-negative, [half + 1, max - 1) for
 * negative. The sentinel value (max) is avoided.
 */
CELER_FORCEINLINE PrimaryId to_primary_id(int track_id)
{
    using size_type = PrimaryId::size_type;
    using common_uint = std::common_type_t<unsigned int, size_type>;

    CELER_EXPECT(static_cast<common_uint>(std::abs(track_id))
                 < static_cast<common_uint>(neg_trackid_offset));

    if (track_id >= 0)
    {
        // Map non-negative track IDs to lower half
        return id_cast<PrimaryId>(track_id);
    }

    // Map negative track IDs to upper half: -1 -> half + 1, -2 -> half+2, etc.
    // Absolute value of track_id should fit in the upper half range
    auto uid = neg_trackid_offset + static_cast<size_type>(-track_id);
    return id_cast<PrimaryId>(uid);
}

//---------------------------------------------------------------------------//
//! Convert Celeritas primary ID to LArSoft track ID.
CELER_FORCEINLINE int to_track_id(PrimaryId primary_id)
{
    CELER_EXPECT(primary_id);

    auto const uid = primary_id.get();

    if (uid < neg_trackid_offset)
    {
        // Lower half: direct conversion to non-negative int
        return static_cast<int>(uid);
    }

    // Upper half: convert to negative int
    // Compute offset from half_range: 0 -> -1, 1 -> -2, etc.
    auto offset = uid - neg_trackid_offset;
    return -static_cast<std::make_signed_t<size_type>>(offset);
}

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

    CELER_LOG(info) << "Setting up Celeritas optical standalone runner "
                       "built against LArSoft v"
                    << cmake::larsoft_version << " components";

    i.problem.detectors.callback
        = [this](SpanCelerHits h) { return this->hit(h); };
    runner_ = std::make_shared<optical::Runner>(std::move(i));

    // Map detector coordinates
    auto geo = runner_->params()->geometry();
    CELER_ASSERT(geo);
    auto vols = runner_->params()->volume();
    CELER_ASSERT(vols);
    auto dets = runner_->params()->detectors();
    CELER_ASSERT(dets);

    channel_to_geo_.resize(det_coords.size());
    btr_helpers_.reserve(det_coords.size());
    for (auto i : range(det_coords.size()))
    {
        auto inst_id = geo->find_volume_instance_at(det_coords[i]);
        CELER_VALIDATE(inst_id,
                       << "could not find a volume at " << det_coords[i]
                       << " [" << lengthunits::native_label << "]");
        channel_to_geo_[i] = inst_id;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Run scintillation optical photons from a single set of energy steps.
 *
 * The optical material is determined in Celeritas when the tracks are
 * initialized from the pre-step position.
 *
 * \todo With Cherenkov enabled we would need to determine the incident
 * particle's charge and the pre- and post-step speed.
 */
auto LarStandaloneRunner::operator()(VecSED const& sim_energy_deposits)
    -> VecBTR
{
    CELER_EXPECT(!sim_energy_deposits.empty());

    // Allocate BTR helpers
    btr_helpers_.clear();
    for (auto i : range(channel_to_geo_.size()))
    {
        auto&& [iter, inserted] = btr_helpers_.emplace(
            channel_to_geo_[i], std::make_unique<sim::OBTRHelper>(i));
        CELER_ASSERT(inserted);
    }

    // Convert SimEnergyDep input
    std::vector<celeritas::optical::GeneratorDistributionData> gdd;
    gdd.reserve(sim_energy_deposits.size());
    size_type num_skipped{0};
    double edep_skipped{0};
    for (auto const& step : sim_energy_deposits)
    {
        if (step.NumPhotons() == 0)
        {
            ++num_skipped;
            edep_skipped += step.E();
            continue;
        }

        // Convert LArSoft sim edeps to Celeritas generator distribution data
        celeritas::optical::GeneratorDistributionData data;
        data.type = GeneratorType::scintillation;
        data.num_photons = step.NumPhotons();
        data.primary = to_primary_id(step.TrackID());
        data.step_length = convert_from_larsoft<LarsoftLen>(step.StepLength());
        // Assume continuous energy loss along the step
        //! \todo For neutral particles, set this to 0 (LED at post-step point)
        data.continuous_edep_fraction = 1;
        data.points[StepPoint::pre].time
            = convert_from_larsoft<LarsoftTime>(step.StartT());
        data.points[StepPoint::pre].pos
            = convert_from_larsoft<LarsoftLen>(step.Start());
        data.points[StepPoint::post].time
            = convert_from_larsoft<LarsoftTime>(step.EndT());
        data.points[StepPoint::post].pos
            = convert_from_larsoft<LarsoftLen>(step.End());
        CELER_ASSERT(data);
        gdd.push_back(data);
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

    // Execute
    auto result = (*runner_)(make_span(std::as_const(gdd)));

    CELER_ASSERT(result.counters.generators.size() == 1);
    auto const& gen = result.counters.generators.front();
    CELER_LOG(debug) << "Transported " << gen.num_generated
                     << " optical photons from " << gen.buffer_size
                     << " sim energy deposits with a total of "
                     << result.counters.steps << " steps over "
                     << result.counters.step_iters << " step iterations";

    // Convert BTR helpers to BTRs in the LarSoft order
    VecBTR btrs;
    btrs.reserve(btr_helpers_.size());
    for (VolumeInstanceId vi : channel_to_geo_)
    {
        auto iter = btr_helpers_.find(vi);
        CELER_ASSERT(iter != btr_helpers_.end());
        CELER_ASSERT(iter->second);
        btrs.emplace_back(make_obtr(std::move(*iter->second)));
    }
    btr_helpers_.clear();

    return btrs;
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
        auto btr_iter = btr_helpers_.find(h.volume_instance);
        CELER_ASSERT(btr_iter != btr_helpers_.end());

        // Note: BTR call requires a pointer, which ROOT doesn't support, so we
        // convert to a native quantity
        auto larpos = value_as<LarsoftLen>(make_quantity_array<LarsoftLen>(
            static_array_cast<double>(h.position)));
        btr_iter->second->AddScintillationPhotonsToMap(
            to_track_id(h.primary),
            convert_to_larsoft<LarsoftTime>(h.time),
            /* num photons = */ 1,
            larpos.data(),
            value_as<units::MevEnergy>(h.energy));
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
