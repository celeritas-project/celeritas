//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoOpticalIdMap.cc
//---------------------------------------------------------------------------//
#include "GeoOpticalIdMap.hh"

#include <G4Material.hh>

#include "corecel/cont/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from underlying Geant4 objects.
 */
GeoOpticalIdMap::GeoOpticalIdMap(G4MaterialTable const& mt)
    : geo_to_opt_{mt.size()}
{
    OptMatId::size_type next_id{0};
    for (auto mat_idx : range(geo_to_opt_.size()))
    {
        G4Material const* material = mt[mat_idx];
        CELER_ASSERT(material);
        CELER_ASSERT(mat_idx == static_cast<std::size_t>(material->GetIndex()));

        // Add optical material properties, if any are present
        if (auto* mpt = material->GetMaterialPropertiesTable())
        {
            if (mpt->GetProperty("RINDEX"))
            {
                geo_to_opt_[mat_idx] = OptMatId{next_id++};
            }
        }
    }

    num_optical_ = next_id;
    if (next_id == 0)
    {
        // No optical materials: clear the array so we're "false"
        geo_to_opt_ = {};
    }
    CELER_ENSURE(this->empty() == (num_optical_ == 0));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
