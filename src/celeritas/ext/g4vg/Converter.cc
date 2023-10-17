//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/g4vg/Converter.cc
//---------------------------------------------------------------------------//
#include "Converter.hh"

#include <iomanip>
#include <iostream>
#include <unordered_set>
#include <G4LogicalVolumeStore.hh>
#include <G4PVPlacement.hh>
#include <G4ReflectionFactory.hh>
#include <G4VPhysicalVolume.hh>
#include <VecGeom/gdml/ReflFactory.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "celeritas/ext/GeantGeoUtils.hh"

#include "LogicalVolumeConverter.hh"
#include "Scaler.hh"
#include "SolidConverter.hh"
#include "Transformer.hh"

namespace celeritas
{
namespace g4vg
{
namespace
{
//---------------------------------------------------------------------------//
//! Get the underlying volume if one exists (const-correct)
G4LogicalVolume const* get_constituent_lv(G4LogicalVolume const& lv)
{
    return G4ReflectionFactory::Instance()->GetConstituentLV(
        const_cast<G4LogicalVolume*>(&lv));
}

//---------------------------------------------------------------------------//
//! Add all visited logical volumes to a set.
struct LVMapVisitor
{
    std::unordered_set<G4LogicalVolume const*>* all_lv;

    void operator()(G4LogicalVolume const* lv)
    {
        CELER_EXPECT(lv);
        if (auto const* unrefl_lv = get_constituent_lv(*lv))
        {
            // Visit underlying instead of reflected
            return (*this)(unrefl_lv);
        }

        // Add this LV
        all_lv->insert(lv);

        // Visit daughters
        for (auto const i : range(lv->GetNoDaughters()))
        {
            (*this)(lv->GetDaughter(i)->GetLogicalVolume());
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
//! Construct with scale
Converter::Converter(Options options)
    : options_{options}
    , convert_scale_{std::make_unique<Scaler>()}
    , convert_transform_{std::make_unique<Transformer>(*convert_scale_)}
    , convert_solid_{std::make_unique<SolidConverter>(
          *convert_scale_, *convert_transform_, options.compare_volumes)}
    , convert_lv_{std::make_unique<LogicalVolumeConverter>(*convert_solid_)}
{
}

//---------------------------------------------------------------------------//
//! Default destructor
Converter::~Converter() = default;

//---------------------------------------------------------------------------//
auto Converter::operator()(arg_type g4world) -> result_type
{
    CELER_EXPECT(g4world);
    CELER_EXPECT(!g4world->GetRotation());
    CELER_EXPECT(g4world->GetTranslation() == G4ThreeVector(0, 0, 0));

    CELER_LOG(status) << "Converting Geant4 geometry";
    ScopedProfiling profile_this{"import-geant-geo"};
    ScopedMem record_mem("Converter.convert");
    ScopedTimeLog scoped_time;

    // Recurse through physical volumes once to build underlying LV
    std::unordered_set<G4LogicalVolume const*> all_g4lv;
    all_g4lv.reserve(G4LogicalVolumeStore::GetInstance()->size());
    LVMapVisitor{&all_g4lv}(g4world->GetLogicalVolume());

    // Convert visited volumes in instance order to try to approximate layout
    // of Geant4
    for (auto* lv : *G4LogicalVolumeStore::GetInstance())
    {
        if (all_g4lv.count(lv))
        {
            (*this->convert_lv_)(*lv);
        }
    }

    // Place world volume
    VGLogicalVolume* world_lv
        = this->build_with_daughters(g4world->GetLogicalVolume());
    auto trans = (*this->convert_transform_)(g4world->GetTranslation(),
                                             g4world->GetRotation());

    result_type result;
    result.world = world_lv->Place(g4world->GetName().c_str(), &trans);
    result.volumes = convert_lv_->make_volume_map();

    CELER_ENSURE(result.world);
    CELER_ENSURE(!result.volumes.empty());
    return result;
}

//---------------------------------------------------------------------------//
//! \cond
/*!
 * Convert a volume and its daughter volumes.
 */
auto Converter::build_with_daughters(G4LogicalVolume const* mother_g4lv)
    -> VGLogicalVolume*
{
    CELER_EXPECT(mother_g4lv);

    if (CELER_UNLIKELY(options_.verbose))
    {
        std::clog << std::string(depth_, ' ') << "Converting "
                  << mother_g4lv->GetName() << std::endl;
    }

    // Convert or get corresponding VecGeom volume
    VGLogicalVolume* mother_lv = (*this->convert_lv_)(*mother_g4lv);

    if (auto [iter, inserted] = built_daughters_.insert(mother_lv); !inserted)
    {
        // Daughters have already been built
        return mother_lv;
    }

    ++depth_;

    // Place daughter logical volumes in this mother
    for (auto const i : range(mother_g4lv->GetNoDaughters()))
    {
        // Get daughter volume
        G4VPhysicalVolume const* g4pv = mother_g4lv->GetDaughter(i);
        G4LogicalVolume const* g4lv = g4pv->GetLogicalVolume();
        if (!dynamic_cast<G4PVPlacement const*>(g4pv))
        {
            TypeDemangler<G4VPhysicalVolume> demangle_pv_type;
            CELER_LOG(error)
                << "Unsupported type '" << demangle_pv_type(*g4pv)
                << "' for physical volume '" << g4pv->GetName()
                << "' (corresponding LV: " << PrintableLV{g4lv} << ")";
        }

        // Test for reflection
        bool flip_z = false;
        if (G4LogicalVolume const* unrefl_g4lv = get_constituent_lv(*g4lv))
        {
            // Replace with constituent volume, and flip the Z scale
            // See G4ReflectionFactory::CheckScale: the reflection value is
            // hard coded to {1, 1, -1}
            g4lv = unrefl_g4lv;
            flip_z = true;
        }

        // Convert daughter volume
        VGLogicalVolume* daughter_lv = this->build_with_daughters(g4lv);

        // Use the VGDML reflection factory to place the daughter in the mother
        // (it must *always* be used, in case parent is reflected)
        vgdml::ReflFactory::Instance().Place(
            (*this->convert_transform_)(g4pv->GetTranslation(),
                                        g4pv->GetRotation()),
            vecgeom::Vector3D<double>{1.0, 1.0, flip_z ? -1.0 : 1.0},
            g4pv->GetName(),
            daughter_lv,
            mother_lv,
            g4pv->GetCopyNo());
    }

    --depth_;

    return mother_lv;
}
//! \endcond

//---------------------------------------------------------------------------//
}  // namespace g4vg
}  // namespace celeritas
