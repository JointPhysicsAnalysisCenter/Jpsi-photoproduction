# BUILD LIBRARY FROM LOCAL FiLES FOR THE MAIN LIBRARY
file(GLOB SRC "core/*.cpp" "covariants/*.cpp" "inclusive/*.cpp" "cubature/*.c" "analysis/*.cpp") 

include_directories(core)
include_directories(covariants)
include_directories(inclusive)
include_directories(cubature)
include_directories(analysis)

ROOT_GENERATE_DICTIONARY(G__JPACPHOTO
    amplitude.hpp
    partial_wave.hpp
    kinematics.hpp
    dirac_spinor.hpp
    dirac_matrix.hpp
    lorentz_tensor.hpp
    fitter.hpp
    data_set.hpp
    box.hpp
    inclusive_process.hpp
    LINKDEF JPACPHOTOLinkDef.h
)
add_library(JPACPHOTO SHARED ${SRC} G__JPACPHOTO.cxx )

target_link_libraries( JPACPHOTO ROOT::Core ROOT::MathMore )
target_link_libraries( JPACPHOTO Boost::boost)

install(TARGETS JPACPHOTO
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} )

install (FILES  ${CMAKE_CURRENT_BINARY_DIR}/libJPACPHOTO_rdict.pcm    
         DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Build the executable to integrate with root cling
add_executable( jpacPhoto "cling/jpacPhoto.cpp" )
target_link_libraries( jpacPhoto ROOT::Core ROOT::MathMore ROOT::RIO ROOT::Rint )
target_link_libraries( jpacPhoto JPACPHOTO )