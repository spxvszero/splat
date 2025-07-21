import * as THREE from 'three';
import CameraControls from 'camera-controls';

CameraControls.install( { THREE: THREE } );

if (!WebGL.isWebGLAvailable() ) {
    const warning = WebGL.getWebGLErrorMessage();
	document.getElementById( 'container' ).appendChild( warning );
    throw new Error("WebGL is not supported");
}

let SCREEN_HEIGHT = window.innerHeight;
let SCREEN_WIDTH = window.innerWidth;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 60, window.innerWidth / window.innerHeight, 0.1, 100 );
const thirdFollowCamParams = new THREE.Vector3( 15, -20, 30)
camera.position.set(thirdFollowCamParams.x, thirdFollowCamParams.y, thirdFollowCamParams.z);
camera.lookAt(0, 0, 0);
camera.up = new THREE.Vector3(0, -1, 0);
window.addEventListener( 'resize', onWindowResize );

const renderer = new THREE.WebGLRenderer({antialias : true});
console.log("webgl render version: ", renderer.webglVersion);
renderer.setSize( window.innerWidth, window.innerHeight );
renderer.setClearColor(0x000000, 0);
renderer.state.setCullFace(THREE.CullFaceNone);
renderer.autoClear = false;

document.body.appendChild( renderer.domElement );
const gl = renderer.getContext();
console.log("wegb is gl2? ", gl instanceof WebGL2RenderingContext);


const fovRad = THREE.MathUtils.degToRad(camera.fov / 2);
const camFy = (window.innerHeight / 2) / Math.tan(fovRad);
const camFx = camFy * camera.aspect;
const splatModel = new SplatRender(gl, camFx, camFy, window.innerWidth, window.innerHeight, camera.projectionMatrix.elements);

//Cube Render
const geometry = new THREE.BoxGeometry( 0.01, 0.01, 0.01 );
const material = new THREE.MeshBasicMaterial( { color: 0x000000, visible: false } );
const cameraCube = new THREE.Mesh( geometry, material );
cameraCube.position.set(0, 0, 0);
// cube.visible = true
scene.add( cameraCube );

cameraCube.add(camera)

addOrbitControl();
function addOrbitControl() {
    controls = new CameraControls( camera, renderer.domElement );
    controls.minDistance = 0.01;
    controls.maxDistance = 100;
    controls.truckSpeed = 0;
    controls.saveState();

}


animate();

function animate(now) {

    const delta = clock.getDelta();
    const updated = controls.update( delta );
    
    const projectionMatrix = camera.projectionMatrix.elements;
    const viewMatrix = camera.matrixWorldInverse.elements;

	requestAnimationFrame( animate );

    camera.lookAt(cameraCube.position)

    if(stats != undefined) {
        stats.begin();
    }

    splatModel.render(now, projectionMatrix, viewMatrix);
        
    // if (updated) {
        renderer.render( scene, camera );
    // }


    if(stats != undefined) {
        stats.end();
    }
}

function onWindowResize() {

    SCREEN_HEIGHT = window.innerHeight;
    SCREEN_WIDTH = window.innerWidth;

    camera.aspect = SCREEN_WIDTH / SCREEN_HEIGHT;
    camera.updateProjectionMatrix();

    renderer.setSize( SCREEN_WIDTH, SCREEN_HEIGHT );

    //splat model
    const fovRad = THREE.MathUtils.degToRad(camera.fov / 2);
    const camFy = (window.innerHeight / 2) / Math.tan(fovRad);
    const camFx = camFy * camera.aspect;
    const projectionMatrix = camera.projectionMatrix;
    splatModel.resize(camFx, camFy, SCREEN_WIDTH, SCREEN_HEIGHT, projectionMatrix.elements);

}


function updatePixelRadio(radio) {
    renderer.setPixelRatio(radio);
}

updatePixelRadio(window.devicePixelRatio);