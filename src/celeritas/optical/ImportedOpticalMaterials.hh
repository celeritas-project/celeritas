//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedOpticalMaterials.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
struct ImportData;
//---------------------------------------------------------------------------//
/*!
 * A registry for imported optical material data.
 */
class ImportedOpticalMaterials
{
  public:
    //! Construct with imported data
    static std::shared_ptr<ImportedOpticalMaterials> from_import(ImportData const& data);

    //! Construct with imported tables
    explicit ImportedOpticalMaterials(std::vector<ImportOpticalMaterial> io);

    //! Get the optical material properties for the given ID
    inline ImportOpticalMaterial const& get(OpticalMaterialId id) const;

    //! Number of imported optical materials
    inline OpticalMaterialId::size_type size() const;

  private:
    std::vector<ImportOpticalMaterial> materials_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the imported optical material data associated with the given optical
 * material ID.
 */
ImportOpticalMaterial const& ImportedOpticalMaterials::get(OpticalMaterialId id) const
{
    CELER_EXPECT(id && id < this->size());
    return materials_[id.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Return the number of optical materials.
 */
OpticalMaterialId::size_type ImportedOpticalMaterials::size() const
{
    return materials_.size();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
