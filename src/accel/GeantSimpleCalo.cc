//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/GeantSimpleCalo.cc
//---------------------------------------------------------------------------//
#include "GeantSimpleCalo.hh"

#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/LabelIO.json.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/GeantUtils.hh"

#include "SharedParams.hh"

#include "detail/GeantSimpleCaloSD.hh"
#include "detail/GeantSimpleCaloStorage.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Sensitive detector present on manager thread (never used)
class DummyGeantSimpleCaloSD : public G4VSensitiveDetector
{
  public:
    // Construct with name and shared params
    DummyGeantSimpleCaloSD(std::string const& name)
        : G4VSensitiveDetector{name}
    {
    }

  protected:
    void Initialize(G4HCofThisEvent*) final {}
    bool ProcessHits(G4Step*, G4TouchableHistory*) final
    {
        CELER_ASSERT_UNREACHABLE();
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with name and logical volume pointers.
 */
GeantSimpleCalo::GeantSimpleCalo(std::string name,
                                 SPConstParams params,
                                 VecLV volumes)
    : params_{std::move(params)}
    , volumes_{std::move(volumes)}
    , storage_{std::make_shared<detail::GeantSimpleCaloStorage>()}
{
    CELER_EXPECT(!name.empty());
    CELER_EXPECT(params_);
    CELER_EXPECT(!volumes_.empty());

    storage_->name = std::move(name);

    // Create LV map
    storage_->volume_to_index.reserve(volumes_.size());
    for (auto i : range(volumes_.size()))
    {
        CELER_EXPECT(volumes_[i]);
        auto&& [iter, inserted]
            = storage_->volume_to_index.insert({volumes_[i], i});
        CELER_VALIDATE(inserted,
                       << "logical volume " << PrintableLV{iter->first}
                       << " is duplicated in the list of volumes for "
                          "GeantSimpleCalo '"
                       << this->label() << "'");
    }

    // Resize data
    storage_->num_threads = params_->num_streams();
    storage_->data.resize(storage_->num_threads);

    CELER_ENSURE(!storage_->name.empty());
    CELER_ENSURE(storage_->volume_to_index.size() == volumes_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Emit a new detector for the local thread and attach to the stored LVs.
 */
auto GeantSimpleCalo::MakeSensitiveDetector() -> UPSensitiveDetector
{
    UPSensitiveDetector detector;

    // Get thread ID
    auto thread_id = get_geant_thread_id();
    if (thread_id < 0)
    {
        // Manager thread: use a dummy SD
        detector = std::make_unique<DummyGeantSimpleCaloSD>(storage_->name);
    }
    else
    {
        CELER_ASSERT(static_cast<size_type>(thread_id) < storage_->num_threads);
        CELER_VALIDATE(storage_->data[thread_id].empty(),
                       << "tried to create multiple SDs for thread "
                       << thread_id << " of simple calo '" << storage_->name
                       << "'");

        // Allocate and clear result
        storage_->data[thread_id].assign(storage_->volume_to_index.size(), 0.0);

        // Create SD
        detector
            = std::make_unique<detail::GeantSimpleCaloSD>(storage_, thread_id);
    }

    // Attach SD to LVs
    for (auto const& lv_idx : storage_->volume_to_index)
    {
        CELER_LOG(debug) << "Attaching '" << storage_->name << "'@"
                         << detector.get() << " to '"
                         << lv_idx.first->GetName() << "'@"
                         << static_cast<void const*>(lv_idx.first);
        lv_idx.first->SetSensitiveDetector(detector.get());
    }
    return detector;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate thread-integrated energy deposition.
 *
 * This function should only be called after all detector data has been
 * collected.
 */
auto GeantSimpleCalo::CalcTotalEnergyDeposition() const -> VecReal
{
    VecReal result(volumes_.size(), 0.0);
    if (storage_->data.empty())
    {
        CELER_LOG(warning) << "No SDs were created from GeantSimpleCalo '"
                           << this->label() << "'";
    }

    for (auto thread_id : range(storage_->data.size()))
    {
        VecReal const& thread_data = storage_->data[thread_id];
        if (thread_data.empty())
        {
            CELER_LOG(warning)
                << "No SD was emitted from GeantSimpleCalo '" << this->label()
                << "' for thread index " << thread_id;
            continue;
        }

        for (auto vol_idx : range(thread_data.size()))
        {
            result[vol_idx] += thread_data[vol_idx];
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Return the key in the JSON output.
 */
std::string_view GeantSimpleCalo::label() const
{
    return storage_->name;
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void GeantSimpleCalo::output(JsonPimpl* j) const
{
    using json = nlohmann::json;

    auto obj = json::object();

    // Save detector volumes
    {
        auto const* geo = celeritas::geant_geo();
        std::shared_ptr<GeantGeoParams> ggp;
        if (!geo)
        {
            // This can happen if using this class without Celeritas offloading
            // enabled, i.e. CELER_DISABLE=1
            ggp = GeantGeoParams::from_tracking_manager();
            geo = ggp.get();
        }
        std::vector<int> ids(volumes_.size());
        std::vector<Label> labels(volumes_.size());

        for (auto idx : range(volumes_.size()))
        {
            auto id = geo->find_volume(volumes_[idx]);
            if (id)
            {
                ids[idx] = id.unchecked_get();
                labels[idx] = geo->volumes().at(id);
            }
        }
        obj["volume_ids"] = std::move(ids);
        obj["volume_labels"] = std::move(labels);
    }

    // Save results
    {
        obj["energy_deposition"] = this->CalcTotalEnergyDeposition();
        obj["_units"] = {
            {"energy_deposition", EnergyUnits::label()},
        };
    }

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
