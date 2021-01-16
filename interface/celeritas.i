//---------------------------------*-SWIG-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file interface/celeritas.i
//---------------------------------------------------------------------------//
%module "celeritas"

%include <cstring.i>
%feature("flatnested");

//---------------------------------------------------------------------------//
// CONFIG FILE
//---------------------------------------------------------------------------//
%{
#include "celeritas_config.h"
%}

%include "celeritas_config.h"

//---------------------------------------------------------------------------//
// BASE/ASSERT
//---------------------------------------------------------------------------//

%{
#include <stdexcept>
%}

%include <exception.i>

%exception {
  try { $action }
  catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

//---------------------------------------------------------------------------//
// BASE/TYPES
//---------------------------------------------------------------------------//

%{
#include "base/Types.hh"
%}

%ignore celeritas::Byte;
%include "base/Types.hh"

%include <std_vector.i>
%template(VecReal) std::vector<celeritas::real_type>;

//---------------------------------------------------------------------------//
// BASE/UNITS
//---------------------------------------------------------------------------//

%{
#include "base/Units.hh"
%}

%include "base/Units.hh"

//---------------------------------------------------------------------------//
// BASE/CONSTANTS
//---------------------------------------------------------------------------//

%{
#include "base/Constants.hh"
%}

%include "base/Constants.hh"

//---------------------------------------------------------------------------//
// PHYSICS
//---------------------------------------------------------------------------//

namespace celeritas
{
class ParticleParams;
class MaterialParams;
}

//---------------------------------------------------------------------------//
// IO
//---------------------------------------------------------------------------//

%{
#include "io/RootImporter.hh"
%}

// TODO: all the `to_cstrings` overload on enum type, but all enums look the
// same in Python.
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) to_cstring;

%include "io/ImportPhysicsVector.hh"
%template(VecImportPhysicsVector) std::vector<celeritas::ImportPhysicsVector>;

%rename(xs_lo) celeritas::ImportTableType::lambda;
%rename(xs_hi) celeritas::ImportTableType::lambda_prim;
%include "io/ImportPhysicsTable.hh"

%template(VecImportPhysicsTable) std::vector<celeritas::ImportPhysicsTable>;
%include "io/ImportProcess.hh"

// %include "io/GdmlGeometryMapTypes.hh"
// %include "io/GdmlGeometryMap.hh"
namespace celeritas
{
class GdmlGeometryMap;
}

%rename(RootImportResult) celeritas::RootImporter::result_type;
%include "io/RootImporter.hh"

// vim: set ft=lex ts=2 sw=2 sts=2 :
