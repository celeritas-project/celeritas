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
    std::deque<QueuedDaughter> child_queue;

    // Convert a physical volume, queuing children if needed
    PhysicalVolume make_pv(int depth, G4VPhysicalVolume const& pv);

    // Build a child
    void
    place_child(int depth, G4VPhysicalVolume const& g4pv, LogicalVolume* lv);

    // Build all children in the queue
    void build_children();
};

//---------------------------------------------------------------------------//
/*!
 * Construct with options.
 */
PhysicalVolumeConverter::PhysicalVolumeConverter(Options opts)
    : data_{std::make_unique<Data>()}
{
    data_->scale = Scaler{opts.scale};
    data_->verbose = true;  // opts.verbose;
}

//---------------------------------------------------------------------------//
//! Default destructor
PhysicalVolumeConverter::~PhysicalVolumeConverter() = default;

//---------------------------------------------------------------------------//
auto PhysicalVolumeConverter::operator()(arg_type g4world) -> result_type
{
    CELER_EXPECT(!g4world.GetRotation());
    CELER_EXPECT(g4world.GetTranslation() == G4ThreeVector(0, 0, 0));

    ScopedProfiling profile_this{"import-geant-geo"};
    ScopedMem record_mem("PhysicalVolumeConverter.convert");
    ScopedTimeLog scoped_time;

    // Construct world volume
    Builder impl{data_.get(), {}};
    auto world = impl.make_pv(0, g4world);
    impl.build_children();
    return world;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a physical volume without building children.
 */
PhysicalVolume
PhysicalVolumeConverter::Builder::make_pv(int depth,
                                          G4VPhysicalVolume const& g4pv)
{
    PhysicalVolume result;

    result.name = g4pv.GetName();
    result.copy_number = g4pv.GetCopyNo();
    result.transform = [&]() -> VariantTransform {
        auto const& g4trans = g4pv.GetObjectTranslation();
        if (g4pv.GetFrameRotation())
        {
            // Get the child-to-parent rotation and check for being identity
            // (parameterized volumes inject an identity matrix)
            auto const& rot = g4pv.GetObjectRotationValue();
            if (!rot.isIdentity())
            {
                return this->data->make_transform(g4trans, rot);
            }
        }
        if (g4trans[0] != 0 || g4trans[1] != 0 || g4trans[2] != 0)
        {
            return this->data->make_transform(g4pv.GetObjectTranslation());
        }
        return NoTransformation{};
    }();

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
    auto&& [lv, inserted] = this->data->make_lv(*g4lv);
    if (inserted)
    {
        if (CELER_UNLIKELY(data->verbose))
        {
            std::clog << std::string(depth, ' ') << "Converted "
                      << g4lv->GetName() << std::endl;
        }
        // Queue up children for construction
        auto num_children = g4lv->GetNoDaughters();
        lv->children.reserve(num_children);
        for (auto i : range(num_children))
        {
            G4VPhysicalVolume* g4pv = g4lv->GetDaughter(i);
            CELER_ASSERT(g4pv);
            child_queue.push_back({depth + 1, g4pv, lv});
        }
    }
    result.lv = std::move(lv);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Convert physical volumes that belong to a Geant4 LV.
 *
 * This adds the result to the ORANGE LogicalVolume.
 */
void PhysicalVolumeConverter::Builder::place_child(
    int depth, G4VPhysicalVolume const& g4pv, LogicalVolume* lv)
{
    if (dynamic_cast<G4PVPlacement const*>(&g4pv))
    {
        // Place child, accounting for reflection
        lv->children.push_back(this->make_pv(depth, g4pv));
    }
    else if (G4VPVParameterisation* param = g4pv.GetParameterisation())
    {
        if (CELER_UNLIKELY(data->verbose))
        {
            CELER_LOG(debug)
                << "Processing parameterized volume " << g4pv.GetName()
                << " with " << g4pv.GetMultiplicity() << " instances";
        }

        // Loop over number of replicas
        for (auto j : range(g4pv.GetMultiplicity()))
        {
            // Use the parameterization to *change* the physical volume's
            // position (yes, this is how Geant4 does it too)
            param->ComputeTransformation(
                j, const_cast<G4VPhysicalVolume*>(&g4pv));

            // Add a copy
            lv->children.push_back(this->make_pv(depth, g4pv));
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
 * Construct children in a logical volume.
 */
void PhysicalVolumeConverter::Builder::build_children()
{
    while (!child_queue.empty())
    {
        // Grab the front item and pop from the stack
        auto front = std::move(child_queue.front());
        child_queue.pop_front();

        // Build children, potentially queueing more children
        CELER_ASSERT(front.g4pv && front.lv);
        this->place_child(front.depth, *front.g4pv, front.lv.get());
    }
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
