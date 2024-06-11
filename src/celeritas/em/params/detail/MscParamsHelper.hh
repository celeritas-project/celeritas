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
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/CommonCoulombData.hh"
#include "celeritas/io/ImportModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleParams;
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
    using Energy = units::MevEnergy;
    using EnergyBounds = Array<Energy, 2>;
    using VecImportMscModel = std::vector<ImportMscModel>;
    using XsValues = Collection<XsGridData, Ownership::value, MemSpace::host>;
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;
    //!@}

    MscParamsHelper(ParticleParams const&,
                    VecImportMscModel const&,
                    ImportModelClass);

    void build_ids(CoulombIds* ids) const;
    void build_xs(XsValues*, Values*) const;
    EnergyBounds energy_grid_bounds() const;

  private:
    //// DATA ////

    ParticleParams const& particles_;
    ImportModelClass model_class_;
    Array<ParticleId, 2> par_ids_;
    Array<ImportPhysicsTable const*, 2> xs_tables_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
