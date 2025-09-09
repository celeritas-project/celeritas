//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include <G4ReplicaNavigation.hh>
#include <G4VPVParameterisation.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/io/StreamableVariant.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "geocel/GeantGeoParams.hh"
#include "orange/transform/TransformIO.hh"

#include "LogicalVolumeConverter.hh"
#include "Scaler.hh"
#include "SolidConverter.hh"
#include "Transformer.hh"

namespace celeritas
{
namespace g4org
{
namespace
{
//---------------------------------------------------------------------------//
// See G4Navigator::LocateGlobalPointAndSetup for implementation of these
struct ReplicaUpdater
{
    void operator()(int copy_no, G4VPhysicalVolume& g4pv)
    {
        nav_.ComputeTransformation(copy_no, &g4pv);
        g4pv.SetCopyNo(copy_no);
    }

    G4ReplicaNavigation nav_;
};

struct ParamUpdater
{
    void operator()(int copy_no, G4VPhysicalVolume& g4pv)
    {
        // TODO: this only works with parameterized transformations, not
        // changes to the solid or material.
        param_.ComputeTransformation(copy_no, &g4pv);
        g4pv.SetCopyNo(copy_no);
    }

    G4VPVParameterisation& param_;
};

}  // namespace

//---------------------------------------------------------------------------//
struct PhysicalVolumeConverter::Data
{
    explicit Data(GeantGeoParams const& g) : geo{g} {}

    // Geometry pointer
    GeantGeoParams const& geo;
    // Scale with CLHEP units (mm)
    Scaler scale;
    // Transform using the scale
    Transformer make_transform{scale};
    // Convert a solid/shape
    SolidConverter make_solid{scale, make_transform};
    // Convert and cache a logical volume
    LogicalVolumeConverter make_lv{geo, make_solid};
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
PhysicalVolumeConverter::PhysicalVolumeConverter(GeantGeoParams const& geo,
                                                 Options opts)
    : data_{std::make_unique<Data>(geo)}
{
    data_->scale = Scaler{opts.scale};
    data_->verbose = opts.verbose;
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
    ScopedMem record_mem("orange.convert-geant");

    CELER_LOG(status) << "Converting Geant4 geometry elements to ORANGE input";
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

    // Get PV ID, using embedded copy number
    result.id = this->data->geo.geant_to_id(g4pv);

    // Get transform
    result.transform = [&]() -> VariantTransform {
        auto const& g4trans = g4pv.GetObjectTranslation();
        if (g4pv.GetFrameRotation())
        {
            // Get the child-to-parent rotation and do another check for the
            // identity matrix (parameterized volumes often have one)
            auto const& rot = g4pv.GetObjectRotationValue();
            if (!rot.isIdentity())
            {
                return this->data->make_transform(g4trans, rot);
            }
        }
        if (g4trans[0] != 0 || g4trans[1] != 0 || g4trans[2] != 0)
        {
            return this->data->make_transform(g4trans);
        }
        return NoTransformation{};
    }();

    // Convert logical volume
    auto* g4lv = g4pv.GetLogicalVolume();
    auto&& [lv, inserted] = this->data->make_lv(*g4lv);
    if (inserted)
    {
        if (CELER_UNLIKELY(data->verbose))
        {
            std::clog << std::string(depth, ' ') << "Converted "
                      << g4lv->GetName() << " with transform "
                      << StreamableVariant{result.transform} << std::endl;
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
    auto place_single = [&] {
        // Place child
        lv->children.push_back(this->make_pv(depth, g4pv));
    };
    auto place_multiple = [&](auto&& update_pv) {
        for (auto copy_no : range(g4pv.GetMultiplicity()))
        {
            // Modify the volume's position/size/orientation in-place
            update_pv(copy_no, const_cast<G4VPhysicalVolume&>(g4pv));
            // Place the copy: note that this uses the "updated" PV's state
            lv->children.push_back(this->make_pv(depth, g4pv));
        }
    };

    switch (g4pv.VolumeType())
    {
        case EVolume::kNormal:
            // Place daughter once
            place_single();
            break;
        case EVolume::kReplica:
            // Place daughter in each replicated location
            place_multiple(ReplicaUpdater{});
            break;
        case EVolume::kParameterised:
            // Place each parameterized instance of the daughter
            CELER_ASSERT(g4pv.GetParameterisation());
            place_multiple(ParamUpdater{*g4pv.GetParameterisation()});
            break;
        default:
            CELER_LOG(error) << "Unsupported type '"
                             << TypeDemangler<G4VPhysicalVolume>{}(g4pv)
                             << "' for physical volume '" << g4pv.GetName()
                             << "' (corresponding LV: "
                             << PrintableLV{g4pv.GetLogicalVolume()} << ")";
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
