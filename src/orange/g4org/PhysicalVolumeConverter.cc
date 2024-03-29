//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/PhysicalVolumeConverter.cc
//---------------------------------------------------------------------------//
#include "PhysicalVolumeConverter.hh"

#include <deque>
#include <iomanip>
#include <iostream>
#include <unordered_set>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4ReflectionFactory.hh>
#include <G4VPVParameterisation.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "geocel/GeantGeoUtils.hh"

#include "LogicalVolumeConverter.hh"
#include "Scaler.hh"
#include "SolidConverter.hh"
#include "Transformer.hh"

namespace celeritas
{
namespace g4org
{
//---------------------------------------------------------------------------//
struct PhysicalVolumeConverter::Data
{
    // Scale with CLHEP units (mm)
    Scaler scale;
    // Transform using the scale
    Transformer make_transform{scale};
    // Convert a solid/shape
    SolidConverter make_solid{scale, make_transform};
    // Convert and cache a logical volume
    LogicalVolumeConverter make_lv{make_solid};
    // Whether to print extra output
    bool verbose{false};
};

struct PhysicalVolumeConverter::Builder
{
    struct QueuedDaughter
    {
        int depth{};
        G4VPhysicalVolume const* g4pv{nullptr};
        std::shared_ptr<LogicalVolume> lv;
    };

    Data* data;
    std::deque<QueuedDaughter> daughter_queue;

    // Convert a physical volume, queuing daughters if needed
    PhysicalVolume make_pv(int depth, G4VPhysicalVolume const& pv);

    // Build a daughter
    void
    place_daughter(int depth, G4VPhysicalVolume const& g4pv, LogicalVolume* lv);

    // Build all daughters in the queue
    void build_daughters();
};

//---------------------------------------------------------------------------//
/*!
 * Construct with options.
 */
PhysicalVolumeConverter::PhysicalVolumeConverter(bool verbose)
    : data_{std::make_unique<Data>()}
{
    data_->verbose = verbose;
}

//---------------------------------------------------------------------------//
//! Default destructor
PhysicalVolumeConverter::~PhysicalVolumeConverter() = default;

//---------------------------------------------------------------------------//
auto PhysicalVolumeConverter::operator()(arg_type g4world) -> result_type
{
    CELER_EXPECT(!g4world.GetRotation());
    CELER_EXPECT(g4world.GetTranslation() == G4ThreeVector(0, 0, 0));

    CELER_LOG(status) << "Converting Geant4 geometry";
    ScopedProfiling profile_this{"import-geant-geo"};
    ScopedMem record_mem("PhysicalVolumeConverter.convert");
    ScopedTimeLog scoped_time;

    // Construct world volume
    Builder impl{data_.get(), {}};
    auto world = impl.make_pv(0, g4world);
    impl.build_daughters();
    return world;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a physical volume without building daughters.
 */
PhysicalVolume
PhysicalVolumeConverter::Builder::make_pv(int depth,
                                          G4VPhysicalVolume const& g4pv)
{
    PhysicalVolume result;

    result.name = g4pv.GetName();
    result.copy_number = g4pv.GetCopyNo();
    result.transform = this->data->make_transform(g4pv.GetFrameTranslation(),
                                                  g4pv.GetFrameRotation());

    auto* g4lv = g4pv.GetLogicalVolume();
    if (auto* unrefl_g4lv
        = G4ReflectionFactory::Instance()->GetConstituentLV(g4lv))
    {
        // Replace with constituent volume, and reflect across Z.
        // See G4ReflectionFactory::CheckScale: the reflection value is
        // hardcoded to {1, 1, -1}
        g4lv = unrefl_g4lv;
        CELER_NOT_IMPLEMENTED("reflecting a placed volume");
    }

    // Convert logical volume
    if (CELER_UNLIKELY(data->verbose))
    {
        std::clog << std::string(depth, ' ') << "Converting "
                  << g4lv->GetName() << std::endl;
    }
    auto&& [lv, inserted] = this->data->make_lv(*g4lv);
    if (inserted)
    {
        // Queue up daughters for construction
        auto num_daughters = g4lv->GetNoDaughters();
        lv->daughters.reserve(num_daughters);
        for (auto i : range(num_daughters))
        {
            G4VPhysicalVolume* g4pv = g4lv->GetDaughter(i);
            CELER_ASSERT(g4pv);
            daughter_queue.push_back({depth + 1, g4pv, lv});
        }
    }
    result.lv = std::move(lv);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct daughters in a logical volume.
 */
void PhysicalVolumeConverter::Builder::place_daughter(
    int depth, G4VPhysicalVolume const& g4pv, LogicalVolume* lv)
{
    if (CELER_UNLIKELY(data->verbose))
    {
        std::clog << std::string(depth, ' ') << "Placing " << g4pv.GetName()
                  << std::endl;
    }

    if (dynamic_cast<G4PVPlacement const*>(&g4pv))
    {
        // Place daughter, accounting for reflection
        lv->daughters.push_back(this->make_pv(depth, g4pv));
    }
    else if (G4VPVParameterisation* param = g4pv.GetParameterisation())
    {
        // Loop over number of replicas
        for (auto j : range(g4pv.GetMultiplicity()))
        {
            // Use the paramterization to *change* the physical volume's
            // position (yes, this is how Geant4 does it too)
            param->ComputeTransformation(
                j, const_cast<G4VPhysicalVolume*>(&g4pv));

            // Add a copy
            lv->daughters.push_back(this->make_pv(depth, g4pv));
        }
    }
    else
    {
        TypeDemangler<G4VPhysicalVolume> demangle_pv_type;
        CELER_LOG(error)
            << "Unsupported type '" << demangle_pv_type(g4pv)
            << "' for physical volume '" << g4pv.GetName()
            << "' (corresponding LV: " << PrintableLV{g4pv.GetLogicalVolume()}
            << ")";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct daughters in a logical volume.
 */
void PhysicalVolumeConverter::Builder::build_daughters()
{
    while (!daughter_queue.empty())
    {
        // Grab the front item and pop from the stack
        auto front = std::move(daughter_queue.front());
        daughter_queue.pop_front();

        // Build daughters, potentially queueing more daughters
        CELER_ASSERT(front.g4pv && front.lv);
        this->place_daughter(front.depth, *front.g4pv, front.lv.get());
    }
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
