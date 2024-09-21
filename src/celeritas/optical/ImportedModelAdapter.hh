//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedModelAdapter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "celeritas/Types.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportPhysicsVector.hh"

#include "Types.hh"

namespace celeritas
{
namespace optical
{
/*!
 * Raw impoted optical model data.
 */
struct ImportOpticalModel
{
    ImportModelClass model_class;
    std::vector<ImportPhysicsVector> mfps;  //!< per optical material MFPs
};

/*!
 * Collection of all imported data needed to build optical models.
 */
class ImportedModels
{
  public:
    //! Construct imported model data from general imported data
    static std::shared_ptr<ImportedModels> from_import(ImportData const&);

    //! Construct imported model data directly
    explicit ImportedModels(std::vector<ImportOpticalModel> models,
                            std::vector<ImportOpticalMaterial> materials);

    //! Get imported model data from the given identifier
    inline ImportOpticalModel const& model(ModelId mid) const
    {
        CELER_EXPECT(mid && mid.get() < models_.size());
        return models_[mid.get()];
    }

    //! Number of imported models
    inline ModelId::size_type num_models() const { return models_.size(); }

    //! Get imported material data from the given identifier
    inline ImportOpticalMaterial const& material(OpticalMaterialId mat_id) const
    {
        CELER_EXPECT(mat_id && mat_id.get() < materials_.size());
        return materials_[mat_id.get()];
    }

    //! Number of imported materials
    inline OpticalMaterialId::size_type num_materials() const
    {
        return materials_.size();
    }

  private:
    std::vector<ImportOpticalModel> models_;
    std::vector<ImportOpticalMaterial> materials_;
};

//---------------------------------------------------------------------------//
/*!
 * For a given optical model, this is an adapter to its imported data.
 *
 * MFP tables take preference in the following order:
 *  1. ImportOpticalModel tables
 *  2. ImportOpticalMaterial tables
 *  3. Tables built by model
 *
 * If the MFP table functions return nullptr, then there is not an
 * associated MFP table, and the builder should move to the next option.
 */
class ImportedModelAdapter
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportedModels const>;
    //!@}

  public:
    ImportedModelAdapter(SPConstImported imported, ModelId mid)
        : model_(mid), imported_(imported)
    {
        CELER_EXPECT(imported_);
        CELER_EXPECT(model_ && model_ < imported_->num_models());
    }

    inline ImportOpticalMaterial const& material(OpticalMaterialId mat) const
    {
        CELER_EXPECT(mat && mat < imported_->num_materials());
        return imported_->material(mat);
    }

    inline OpticalMaterialId::size_type num_materials() const
    {
        return imported_->num_materials();
    }

    inline ImportPhysicsVector const* imported_mfp(OpticalMaterialId mat) const
    {
        CELER_EXPECT(mat && mat < imported_->num_materials());

        ImportPhysicsVector const* mfp = nullptr;

        auto const& model = this->model();
        if (mat.get() < model.mfps.size() && model.mfps[mat.get()])
        {
            mfp = &model.mfps[mat.get()];
        }

        return mfp;
    }

    inline ImportPhysicsVector const* material_mfp(OpticalMaterialId mat) const
    {
        switch (this->model().model_class)
        {
            case ImportModelClass::absorption:
                return &this->material(mat).absorption.absorption_length;
            case ImportModelClass::rayleigh:
                return &this->material(mat).rayleigh.mfp;
            default:
                return nullptr;
        }
    }

    inline ImportPhysicsVector const* preferred_mfp(OpticalMaterialId mat) const
    {
        if (auto const* mfp = this->imported_mfp(mat))
        {
            return mfp;
        }
        else if (auto const* mfp = this->material_mfp(mat))
        {
            return mfp;
        }
        else
        {
            return nullptr;
        }
    }

  private:
    inline ImportOpticalModel const& model() const
    {
        return imported_->model(model_);
    }

    ModelId model_;
    SPConstImported imported_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
