//---------------------------------*-SWIG-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas.i
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
// ASSERTIONS
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
// CORECEL
//---------------------------------------------------------------------------//

%{
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
%}

%ignore celeritas::Byte;
%include "corecel/Macros.hh"
%include "corecel/Types.hh"

%include <std_vector.i>
%template(VecReal) std::vector<celeritas::real_type>;

//---------------------------------------------------------------------------//
// CELERITAS
//---------------------------------------------------------------------------//

%{
#include "celeritas/Units.hh"
#include "celeritas/Constants.hh"
%}

%include "celeritas/Units.hh"
%include "celeritas/Constants.hh"

//---------------------------------------------------------------------------//

%{
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/PDGNumber.hh"
%}

%include "celeritas/phys/AtomicNumber.hh"
%include "celeritas/phys/PDGNumber.hh"

//---------------------------------------------------------------------------//
// IO
//---------------------------------------------------------------------------//

%{
#include "celeritas/ext/RootImporter.hh"
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

%include "celeritas/io/ImportPhysicsVector.hh"
%template(VecImportPhysicsVector) std::vector<celeritas::ImportPhysicsVector>;

%include "celeritas/io/ImportPhysicsTable.hh"
%template(VecImportPhysicsTable) std::vector<celeritas::ImportPhysicsTable>;

%include "celeritas/io/ImportProcess.hh"
%template(VecImportProcess) std::vector<celeritas::ImportProcess>;

%include "celeritas/io/ImportParticle.hh"
%template(VecImportParticle) std::vector<celeritas::ImportParticle>;

%include "celeritas/io/ImportElement.hh"
%template(VecImportElement) std::vector<celeritas::ImportElement>;

%include "celeritas/io/ImportMaterial.hh"
%template(VecImportMaterial) std::vector<celeritas::ImportMaterial>;

%include "celeritas/io/ImportVolume.hh"
%template(VecImportVolume) std::vector<celeritas::ImportVolume>;

%include "celeritas/io/ImportData.hh"

%rename(RootImportResult) celeritas::RootImporter::result_type;
%include "celeritas/ext/RootImporter.hh"

//---------------------------------------------------------------------------//

%{
#include "celeritas/io/SeltzerBergerReader.hh"
%}

%include "celeritas/io/ImportSBTable.hh"
%include "celeritas/io/SeltzerBergerReader.hh"

//---------------------------------------------------------------------------//

%{
#include "celeritas/io/LivermorePEReader.hh"
%}

%include "celeritas/io/ImportLivermorePE.hh"
%template(VecImportLivermoreSubshell) std::vector<celeritas::ImportLivermoreSubshell>;

%include "celeritas/io/LivermorePEReader.hh"

// vim: set ft=lex ts=2 sw=2 sts=2 :
