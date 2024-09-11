//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportMaterialAdapter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "celeritas/Types.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
class ImportedMaterials
{
  public:
    static std::shared_ptr<ImportedMaterials>
    from_import(ImportData const& data)
    {
        return std::make_shared<ImportedMaterials>(data.optical_materials);
    }

    explicit ImportedMaterials(std::vector<ImportOpticalMaterial> io)
        : optical_materials_(std::move(io))
    {
    }

    inline ImportOpticalMaterial const& get(OpticalMaterialId opt_mat_id) const
    {
        CELER_EXPECT(opt_mat_id && opt_mat_id < this->size());
        return optical_materials_[opt_mat_id.get()];
    }

    inline OpticalMaterialId::size_type size() const
    {
        return optical_materials_.size();
    }

  private:
    std::vector<ImportOpticalMaterial> optical_materials_;
};

/*!
 */
template<ImportModelClass imc>
class ImportMaterialAdapter
{
  public:
    //!@{
    //! \name Type aliases
    using IMC = ImportModelClass;
    using SPConstImported = std::shared_ptr<ImportedMaterials const>;
    //!@}

  public:
    ImportMaterialAdapter(SPConstImported imported) : imported_(imported)
    {
        CELER_EXPECT(imported);
    }

    inline auto get(OpticalMaterialId opt_mat_id) const
    {
        if constexpr (imc == IMC::absorption)
        {
            return imported_->get(opt_mat_id).absorption;
        }
        else if constexpr (imc == IMC::rayleigh)
        {
            return imported_->get(opt_mat_id).rayleigh;
        }
        else if constexpr (imc == IMC::wls)
        {
            return imported_->get(opt_mat_id).wls;
        }
        else
        {
            // not a recognized model class
            return;
        }
    }

    inline OpticalMaterialId::size_type size() const
    {
        return imported_->size();
    }

  private:
    SPConstImported imported_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
