//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/params/detail/MscParamsHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/em/data/MscData.hh"
#include "celeritas/io/ImportModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleParams;
class MaterialParams;
struct XsGridData;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for constructing multiple scattering params.
 */
class MscParamsHelper
{
  public:
    //!@{
    //! \name Type aliases
    using VecImportMscModel = std::vector<ImportMscModel>;
    using XsValues = Collection<XsGridData, Ownership::value, MemSpace::host>;
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;
    //!@}

    MscParamsHelper(ParticleParams const&,
                    MaterialParams const&,
                    VecImportMscModel const&,
                    ImportModelClass);

    void build_ids(MscIds* ids) const;
    void build_xs(XsValues*, Values*) const;

  private:
    //// DATA ////

    ParticleParams const& particles_;
    MaterialParams const& materials_;
    VecImportMscModel const& mdata_;
    ImportModelClass model_class_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
