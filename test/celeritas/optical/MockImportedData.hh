//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MockImportedData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/io/ImportOpticalModel.hh"
#include "celeritas/io/ImportPhysicsVector.hh"
#include "celeritas/optical/ImportedModelAdapter.hh"
#include "celeritas/optical/MfpBuilder.hh"

#include "Test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//
/*!
 * Imported mock optical data.
 *
 * A base class that provides common mock data and functionality for testing
 * optical physics.
 */
class MockImportedData : public ::celeritas::test::Test
{
  protected:
    //!@{
    //! \name Type aliases
    using Grid = GenericGridRecord;
    using GridId = OpaqueId<Grid>;

    using ImportedMfpTable = std::vector<ImportPhysicsVector>;

    using ImportedModelId = typename ImportedModels::ImportedModelId;
    using SPConstImported = std::shared_ptr<ImportedModels const>;

    template<class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;
    //!@}

    //!@{
    //! \name Access mock data
    static std::vector<ImportOpticalModel> const& import_models();
    static std::vector<ImportOpticalMaterial> const& import_materials();

    static ImportedModelId absorption_id();
    static ImportedModelId rayleigh_id();
    static ImportedModelId wls_id();
    //!@}

    //!@{
    //! \name Construct commonly used objects
    SPConstImported create_empty_imported_models() const;
    SPConstImported create_imported_models() const;
    MfpBuilder create_mfp_builder();
    //!@}

    //!@{
    //! \name Check results
    void check_mfp(ImportPhysicsVector const& expected,
                   ImportPhysicsVector const& imported) const;
    void check_built_table(ImportedMfpTable const& expected,
                           ItemRange<Grid> const& table) const;
    //!@}

    //!@{
    //! \name Storage data
    Items<real_type> reals;
    Items<Grid> grids;
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
