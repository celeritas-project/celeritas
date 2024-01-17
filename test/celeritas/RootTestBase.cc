//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/RootTestBase.cc
//---------------------------------------------------------------------------//
#include "RootTestBase.hh"

#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"

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
    static struct
    {
        std::string geometry_basename;
        ImportData imported;
    } i;
    auto geo_basename = this->geometry_basename();
    if (i.geometry_basename != geo_basename)
    {
        ScopedRootErrorHandler scoped_root_error;

        i.geometry_basename = geo_basename;
        std::string root_inp
            = this->test_data_path("celeritas", i.geometry_basename + ".root");

        RootImporter import(root_inp.c_str());
        i.imported = import();

        // Raise an exception if non-fatal errors were encountered
        scoped_root_error.throw_if_errors();
    }
    CELER_ENSURE(!i.imported.materials.empty()
                 && !i.imported.particles.empty());
    return i.imported;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
