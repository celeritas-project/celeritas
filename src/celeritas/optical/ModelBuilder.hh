//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ModelBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "ImportedModelAdapter.hh"

namespace celeritas
{
namespace optical
{
class Model;

//---------------------------------------------------------------------------//
/*!
 * Abstract base class used to build optical models.
 */
struct ModelBuilder
{
    //!@{
    //! \name Type aliases
    using SPModel = std::shared_ptr<Model>;
    //!@}

    virtual SPModel operator()(ActionId id) const = 0;
};

/*!
 * Helper class used to build optical models that only require an action ID
 * and imported data.
 */
template<class M>
class ImportedModelBuilder : public ModelBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using SPModel = std::shared_ptr<Model>;
    //!@}

  public:
    ImportedModelBuilder(ImportedModelAdapter imported) : imported_(imported)
    {
    }

    SPModel operator()(ActionId id) const override
    {
        return std::make_shared<M>(id, imported_);
    }

  private:
    ImportedModelAdapter imported_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
