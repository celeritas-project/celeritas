//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/detail/GeantVolumeInstanceMapper.cc
//---------------------------------------------------------------------------//
#include "GeantVolumeInstanceMapper.hh"

#include <string>
#include <G4LogicalVolume.hh>
#include <G4ReplicaNavigation.hh>
#include <G4VPVParameterisation.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "geocel/VolumeVisitor.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
//! Interface for visiting geometry volumes
class GeantVolumeAccessor final
    : public VolumeAccessorInterface<G4LogicalVolume const*,
                                     G4VPhysicalVolume const*>
{
  public:
    //! Outgoing volume node from an instance
    VolumeRef volume(VolumeInstanceRef parent) final
    {
        CELER_EXPECT(parent);
        auto result = parent->GetLogicalVolume();
        CELER_ENSURE(result);
        return result;
    }

    //! Outgoing instance edges from a volume
    ContainerVolInstRef children(VolumeRef parent) final
    {
        CELER_EXPECT(parent);
        ContainerVolInstRef result;
        for (auto i : range(parent->GetNoDaughters()))
        {
            result.push_back(parent->GetDaughter(i));
        }
        return result;
    }
};
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct IDs from world volume.
 */
GeantVolumeInstanceMapper::GeantVolumeInstanceMapper(G4PV const& world)
{
    // Track volumes that aren't normal/replica/param
    std::vector<std::string> invalid_volumes;

    // Visit (depth-first) volume instances inside the world
    VolumeVisitor visit_vol{GeantVolumeAccessor{}};
    visit_vol(
        [this, &invalid_volumes](G4VPhysicalVolume const* pv, int) {
            CELER_EXPECT(pv);
            auto [iter, inserted] = base_vi_.insert({pv, {}});
            if (!inserted)
            {
                // Already visited
                return false;
            }

            VolumeInstanceId::size_type multiplicity{1};

            // Categorize volume
            switch (pv->VolumeType())
            {
                case EVolume::kNormal:
                    CELER_ASSERT(!pv->IsReplicated());
                    break;
                case EVolume::kReplica:
                    [[fallthrough]];
                case EVolume::kParameterised:
                    CELER_ASSERT(pv->IsReplicated());
                    multiplicity = pv->GetMultiplicity();
                    CELER_LOG(debug)
                        << "Adding " << multiplicity << " copies of "
                        << pv->GetName() << " (volume instance "
                        << g4pv_.size() << ")";
                    break;
                default:
                    invalid_volumes.push_back(pv->GetName());
                    return false;
            }

            // Add start ID to the map
            iter->second = id_cast<VolumeInstanceId>(g4pv_.size());
            // Add copies of the PV to our vector
            g4pv_.resize(g4pv_.size() + multiplicity, pv);
            // Visit children
            return true;
        },
        &world);

    if (!invalid_volumes.empty())
    {
        CELER_LOG(error)
            << "The Geant4 geometry contains unsupported custom volume types "
               "that will be ignored: "
            << join(invalid_volumes.begin(), invalid_volumes.end(), ", ");
    }

    CELER_ENSURE(std::all_of(g4pv_.begin(), g4pv_.end(), Identity{}));
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume instance using the pv and its current state.
 *
 * For replica volumes, this uses its thread-local copy number. On older
 * versions of Geant4 this is uninitialized until a navigator or user
 * queries/updates it.
 *
 * If the given pv is not included in the geometry for some reason (e.g. it's
 * from a "parallel world"), the result is a null \c VolumeInstanceId .
 */
VolumeInstanceId GeantVolumeInstanceMapper::geant_to_id(G4PV const& pv) const
{
    auto iter = base_vi_.find(&pv);
    if (iter == base_vi_.end())
    {
        return {};
    }

    VolumeInstanceId result = iter->second;
    if (pv.IsReplicated())
    {
        int copy_no = pv.GetCopyNo();
        // NOTE: if this line fails, the thread data isn't initialized.
        CELER_ASSERT(copy_no >= 0 && copy_no < pv.GetMultiplicity());
        result = VolumeInstanceId{result.unchecked_get() + copy_no};
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume instance using the pv and a replica number.
 *
 * If the given PV is not a replica, the copy number will be \em ignored.
 * In either case, the volume's state will not be touched.
 *
 * If the pv is not included in the geometry for some reason, the result is a
 * null \c VolumeInstanceId .
 */
VolumeInstanceId
GeantVolumeInstanceMapper::geant_to_id(G4PV const& pv, int copy_no) const
{
    CELER_EXPECT(copy_no >= 0);
    auto iter = base_vi_.find(&pv);
    if (iter == base_vi_.end())
    {
        return {};
    }

    VolumeInstanceId result = iter->second;
    if (pv.IsReplicated())
    {
        CELER_EXPECT(copy_no < pv.GetMultiplicity());

        result = VolumeInstanceId{result.unchecked_get() + copy_no};
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Return and (if replica) update the volume from an instance ID.
 */
auto GeantVolumeInstanceMapper::id_to_geant(VolumeInstanceId vi_id) const
    -> G4PV const&
{
    CELER_EXPECT(vi_id < this->size());

    G4PV const& result = *g4pv_[vi_id.unchecked_get()];
    if (result.IsReplicated())
    {
        this->update_replica(const_cast<G4PV&>(result), vi_id);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the logical volume associated with an instance ID.
 *
 * Used by model construction, this does \em not do any updating of the
 * physical volume.
 */
auto GeantVolumeInstanceMapper::logical_volume(VolumeInstanceId vi_id) const
    -> G4LV const&
{
    CELER_EXPECT(vi_id < this->size());

    G4PV const* pv = g4pv_[vi_id.unchecked_get()];
    CELER_ENSURE(pv->GetLogicalVolume());
    return *pv->GetLogicalVolume();
}

//---------------------------------------------------------------------------//
void GeantVolumeInstanceMapper::update_replica(G4PV& pv,
                                               VolumeInstanceId vi) const
{
    CELER_EXPECT(vi);
    auto base_vi = base_vi_.at(&pv);
    CELER_ASSERT(base_vi <= vi);

    int copy_no = static_cast<int>(vi.unchecked_get())
                  - static_cast<int>(base_vi.unchecked_get());
    CELER_ASSERT(copy_no < pv.GetMultiplicity());

    auto vt = pv.VolumeType();
    if (vt == EVolume::kReplica)
    {
        static G4ThreadLocal G4ReplicaNavigation replica_nav_;
        replica_nav_.ComputeTransformation(copy_no, &pv);
    }
    else if (vt == EVolume::kParameterised)
    {
        auto* param = pv.GetParameterisation();
        CELER_ASSERT(param);
        param->ComputeTransformation(copy_no, &pv);
    }
    else
    {
        CELER_ASSERT_UNREACHABLE();
    }
    pv.SetCopyNo(copy_no);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
