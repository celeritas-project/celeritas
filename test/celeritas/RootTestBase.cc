//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/RootTestBase.cc
//---------------------------------------------------------------------------//
#include "RootTestBase.hh"

#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/setup/Import.hh"

#include "PersistentSP.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Lazily load ROOT data.
 */
auto RootTestBase::imported_data() const -> ImportData const&
{
    static PersistentSP<ImportData const> pid{"import data"};

    std::string const basename{this->gdml_basename()};
    pid.lazy_update(basename, [&] {
        ScopedRootErrorHandler scoped_root_error;

        std::string root_inp = this->test_data_path(
            "celeritas", std::string{basename} + ".root");

        RootImporter import_root(root_inp.c_str());
        auto result = import_root();

        // Raise an exception if non-fatal errors were encountered
        scoped_root_error.throw_if_errors();

        // Delete unwanted data that could cause physics readers to try to load
        this->fixup_data(result);

        // Assume the ROOT file didn't include file-based data
        setup::physics_from(inp::PhysicsFromGeantFiles{}, result);

        return std::make_shared<ImportData>(std::move(result));
    });
    auto const& result = *pid.value();
    CELER_ENSURE(!result.phys_materials.empty()
                 && !result.geo_materials.empty() && !result.particles.empty());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
