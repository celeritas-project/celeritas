//---------------------------------*-SWIG-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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
class CutoffParams;
class ParticleParams;
class MaterialParams;
}

//---------------------------------------------------------------------------//
// IO
//---------------------------------------------------------------------------//

%{
#include "io/RootImporter.hh"
%}

namespace celeritas
{
%rename(table_type_to_string) to_cstring(ImportTableType);
%rename(units_to_string) to_cstring(ImportUnits);
%rename(vector_type_to_string) to_cstring(ImportPhysicsVectorType);
%rename(process_type_to_string) to_cstring(ImportProcessType);
%rename(process_class_to_string) to_cstring(ImportProcessClass);
%rename(model_to_string) to_cstring(ImportModelClass);

%rename(xs_lo) ImportTableType::lambda;
%rename(xs_hi) ImportTableType::lambda_prim;
}

%include "io/ImportPhysicsVector.hh"
%template(VecImportPhysicsVector) std::vector<celeritas::ImportPhysicsVector>;

%include "io/ImportPhysicsTable.hh"
%template(VecImportPhysicsTable) std::vector<celeritas::ImportPhysicsTable>;

%include "io/ImportProcess.hh"
%template(VecImportProcess) std::vector<celeritas::ImportProcess>;

%include "io/ImportParticle.hh"
%template(VecImportParticle) std::vector<celeritas::ImportParticle>;

%include "io/ImportElement.hh"
%template(VecImportElement) std::vector<celeritas::ImportElement>;

%include "io/ImportMaterial.hh"
%template(VecImportMaterial) std::vector<celeritas::ImportMaterial>;

%include "io/ImportVolume.hh"
%template(VecImportVolume) std::vector<celeritas::ImportVolume>;

%include "io/ImportData.hh"

%rename(RootImportResult) celeritas::RootImporter::result_type;
%include "io/RootImporter.hh"

//---------------------------------------------------------------------------//

%{
#include "io/SeltzerBergerReader.hh"
%}

%include "io/ImportSBTable.hh"
%include "io/SeltzerBergerReader.hh"

//---------------------------------------------------------------------------//

%{
#include "io/LivermorePEReader.hh"
%}

%include "io/ImportLivermorePE.hh"
%template(VecImportLivermoreSubshell) std::vector<celeritas::ImportLivermoreSubshell>;

%include "io/LivermorePEReader.hh"

// vim: set ft=lex ts=2 sw=2 sts=2 :
