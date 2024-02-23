//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/MscParams.hh
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

//---------------------------------------------------------------------------//
/*!
 * Interface class for multiple scattering.
 */
class MscParams
{
  public:
    //!@{
    //! \name Type aliases
    using VecImportMscModel = std::vector<ImportMscModel>;
    //!@}

    //! MSC configuration options
    struct Options
    {
        real_type lambda_limit{1 * units::millimeter};  //!< Lambda limit
        real_type geom_fact{2.5};  //!< Geometry factor
        real_type range_fact{0.04};  //!< Range factor for e-/e+
        real_type safety_fact{0.6};  //!< Safety factor
    };

  public:
    // Virtual destructor for polymorphic deletion
    virtual ~MscParams();

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    MscParams() = default;
    CELER_DEFAULT_COPY_MOVE(MscParams);
    //!@}

    //// TYPES ////

    using XsValues = Collection<XsGridData, Ownership::value, MemSpace::host>;
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;

    //// HELPER FUNCTIONS ////

    void build_parameters(MscParameters*, Options const&) const;
    void build_ids(MscIds*, ParticleParams const&) const;
    void build_xs(XsValues*,
                  Values*,
                  ParticleParams const&,
                  MaterialParams const&,
                  VecImportMscModel const&,
                  ImportModelClass) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
