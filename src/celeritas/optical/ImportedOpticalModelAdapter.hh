//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedOpticalModelAdapter.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A registry for imported optical model data.
 *
 * Assigns a unique ImportOpticalModelId to each imported optical model,
 * and associates the ImportOpticalModelClass enum with the corresponding ID.
 */
class ImportOpticalModels
{
  public:
    //!@{
    //! \name Type aliases
    using ImportOpticalModelId = OpaqueId<ImportOpticalModel>;
    using key_type = ImportOpticalModelClass;
    //!@}

  public:
    //! Construct with imported data
    static std::shared_ptr<ImportOpticalModels> from_import(ImportData const& data);

    //! Construct with imported tables
    explicit ImportOpticalModels(std:vector<ImportOpticalModel> io);

    //! Return the optical process ID for the given optical model class
    ImportOpticalModelId find(key_type) const;

    //! Get the table for the given optical model ID
    inline ImportOpticalModel const& get(ImportOpticalModelId id) const;

    //! Number of imported optical models
    inline ImportOpticalModelId::size_type size() const;

  private:
    std::vector<ImportOpticalModel> models_;
    std::map<key_type, ImportOpticalModelId> ids_;
};

//---------------------------------------------------------------------------//
/*!
 * A lightweight adapter for accessing the imported optical model data for
 * a specific model.
 */
class ImportedOpticalModelAdapter
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportOpticalModels const>;
    using StepLimitBuilder = OpticalModel::StepLimitBuilder;
    //!@}

  public:
    //! Construct directly from imported data for a given model
    explicit ImportedOpticalModelAdapter(SPConstImported imported,
                                         ImportedOpticalModelClass model_class);

    //! Construct step limits for the optical model from imported data
    StepLimitBuilder step_limits(OpticalMaterialId id) const;

    //! Get the lambda table for the optical model
    inline ImportPhysicsTable const& get_lambda() const;

    //! Access the imported optical model
    inline ImportOpticalModel const& get_model() const;

  private:
    SPConstImported imported_;
    ImportOpticalModelClass model_class_;
};


//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the imported optical model data associated with the given model id.
 */
ImportOpticalModel const& ImportOpticalModels::get(ImportOpticalModelId id) const
{
    CELER_EXPECT(id < this->size());
    return models_[id.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the number of imported optical models.
 */
auto ImportOpticalModels::size() const -> ImportOpticalModelId::size_type
{
    return models_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get the lambda table for the optical model for this adapter's model class.
 */
ImportPhysicsTable const& ImportedOpticalModelAdapter::get_lambda() const
{
    return get_model().lambda_table;
}

//---------------------------------------------------------------------------//
/*!
 * Get the optical model associated with this adapter's model class.
 */
ImportOpticalModel const& ImportedOpticalModelAdapter::get_model() const
{
    return imported_->get(imported_->find(model_class_));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
